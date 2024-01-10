"""Training and evaluation for score-based generative models. """
import gc
import io
import logging
import csv
import os
import time
from cleanfid import fid
import numpy as np
np.bool = np.bool_
import torch
import wandb
from absl import flags
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
import datasets
import evaluation
import likelihood
from PIL import Image
import losses
import sampling
import sde_lib
# Keep the import below for registering all model definitions
from functools import partial
from minlora import LoRAParametrization, add_lora, apply_to_lora, disable_lora, enable_lora, get_lora_params, merge_lora, name_is_lora, remove_lora, load_multiple_lora, select_lora
from models import ddpm, ncsnpp, ncsnv2
from models import utils as mutils
from models.ema import ExponentialMovingAverage
from utils import restore_checkpoint, save_checkpoint
import shutil

FLAGS = flags.FLAGS


def train(config, workdir):
    """Runs the training pipeline.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """
    # Initialize wandb, with tensorboard sync
    wandb.init(project="rl-df", config=vars(config), sync_tensorboard=True, mode=disabled)

    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    tb_dir = os.path.join(workdir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)

    # Initialize model.
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join(
        workdir, "checkpoints-meta", "checkpoint.pth")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)
    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])

    # Build data iterators
    train_ds, eval_ds, _ = datasets.get_dataset(config,
                                                uniform_dequantization=config.data.uniform_dequantization)

    train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
    eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min,
                            beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min,
                               beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min,
                            sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5
    elif config.training.sde.lower() == 'rectified_flow':
        sde = sde_lib.RectifiedFlow(init_type=config.sampling.init_type,
                                    noise_scale=config.sampling.init_noise_scale, use_ode_sampler=config.sampling.use_ode_sampler)
        sampling_eps = 1e-3
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                       reduce_mean=reduce_mean, continuous=continuous,
                                       likelihood_weighting=likelihood_weighting)
    eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                      reduce_mean=reduce_mean, continuous=continuous,
                                      likelihood_weighting=likelihood_weighting)

    # Building sampling functions
    if config.training.snapshot_sampling:
        sampling_shape = (config.training.batch_size, config.data.num_channels,
                          config.data.image_size, config.data.image_size)
        sampling_fn = sampling.get_sampling_fn(
            config, sde, sampling_shape, inverse_scaler, sampling_eps,dataf=train_ds)

    num_train_steps = config.training.n_iters

    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    logging.info("Starting training loop at step %d." % (initial_step,))

    for step in range(initial_step, num_train_steps + 1):
        # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
        batch = torch.from_numpy(next(train_iter)['image']._numpy()).to(
            config.device).float()
        batch = batch.permute(0, 3, 1, 2)
        batch = scaler(batch)
        # Execute one training step
        loss = train_step_fn(state, batch)
        if step % config.training.log_freq == 0:
            logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))
            writer.add_scalar("training_loss", loss, step)

        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            save_checkpoint(checkpoint_meta_dir, state)

        # Report the loss on an evaluation dataset periodically
        if step % config.training.eval_freq == 0:
            eval_batch = torch.from_numpy(
                next(eval_iter)['image']._numpy()).to(config.device).float()
            eval_batch = eval_batch.permute(0, 3, 1, 2)
            eval_batch = scaler(eval_batch)
            eval_loss = eval_step_fn(state, eval_batch)
            logging.info("step: %d, eval_loss: %.5e" %
                         (step, eval_loss.item()))
            writer.add_scalar("eval_loss", eval_loss.item(), step)

        # Save a checkpoint periodically and generate samples if needed
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
            # Save the checkpoint.
            save_step = step // config.training.snapshot_freq
            save_checkpoint(os.path.join(
                checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

            # Generate and save samples
            if config.training.snapshot_sampling:
                ema.store(score_model.parameters())
                ema.copy_to(score_model.parameters())
                sample, n = sampling_fn(score_model)
                ema.restore(score_model.parameters())
                this_sample_dir = os.path.join(
                    sample_dir, "iter_{}".format(step))
                os.makedirs(this_sample_dir, exist_ok=True)
                nrow = int(np.sqrt(sample.shape[0]))
                image_grid = make_grid(sample, nrow, padding=2)
                sample = np.clip(sample.permute(0, 2, 3, 1).cpu(
                ).numpy() * 255, 0, 255).astype(np.uint8)
                os.makedirs(this_sample_dir, exist_ok=True)

                sample_file_path = os.path.join(this_sample_dir, "sample.np")
                # Writing the numpy array to a binary file
                with open(sample_file_path, "wb") as fout:
                    np.save(fout, sample)

                image_path = os.path.join(this_sample_dir, "sample.png")
                # Saving the image using Pillow
                with open(image_path, "wb") as fout:
                    pil_image = Image.fromarray(image_grid)
                    pil_image.save(fout)

                # Logging the image to WandB
                images = wandb.Image(image_path)
                wandb.log({f'sample_step_{step}': images})


def evaluate(config,
             workdir,
             eval_folder="eval"):
    """Evaluate trained models.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints.
      eval_folder: The subfolder for storing evaluation results. Default to
        "eval"
    """
    wandb.init(project="rl-df", config=vars(config), sync_tensorboard=True, mode="disabled")
    # Create directory to eval_folder
    eval_dir = os.path.join(workdir, eval_folder)
    os.makedirs(eval_dir, exist_ok=True)
    
    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    checkpoint_dir = os.path.join(workdir, "checkpoints")

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min,
                            beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min,
                               beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min,
                            sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5
    elif config.training.sde.lower() == 'rectified_flow':
        sde = sde_lib.RectifiedFlow(init_type=config.sampling.init_type, noise_scale=config.sampling.init_noise_scale, 
                            discretizing_method=config.sampling.discretizing_method, solver=config.sampling.solver,
                            num_steps=config.sampling.sample_N, edm_config=config.sampling.edm,
                            bellman_config=config.sampling.bellman)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'edm':
        sde = sde_lib.EDM(init_type = config.sampling.init_type, discretizing_method=config.sampling.discretizing_method, 
                        solver=config.sampling.solver, num_steps = config.sampling.sample_N, 
                        edm_config=config.sampling.edm,
                        bellman_config=config.sampling.bellman)
        sampling_eps = 1e-3
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Create the one-step evaluation function when loss computation is enabled
    if config.eval.enable_loss:
        optimize_fn = losses.optimization_manager(config)
        continuous = config.training.continuous
        likelihood_weighting = config.training.likelihood_weighting

        reduce_mean = config.training.reduce_mean
        eval_step = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                       reduce_mean=reduce_mean,
                                       continuous=continuous,
                                       likelihood_weighting=likelihood_weighting)

    # Build the likelihood computation function when likelihood is enabled
    if config.eval.enable_bpd:
        train_ds_bpd, eval_ds_bpd, _ = datasets.get_dataset(config,
                                                      uniform_dequantization=True, evaluation=True)
        if config.eval.bpd_dataset.lower() == 'train':
            ds_bpd = train_ds_bpd
            bpd_num_repeats = 1
        elif config.eval.bpd_dataset.lower() == 'test':
            # Go over the dataset 5 times when computing likelihood on the test dataset
            ds_bpd = eval_ds_bpd
            bpd_num_repeats = 5
        else:
            raise ValueError(f"No bpd dataset {config.eval.bpd_dataset} recognized.")
        if config.training.sde.lower() == 'mixup':
            likelihood_fn = likelihood.get_likelihood_fn_rf(
                sde, inverse_scaler)
        else:
            likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler)

    begin_ckpt = config.eval.begin_ckpt
    logging.info("begin checkpoint: %d" % (begin_ckpt,))
    for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
        # Wait if the target checkpoint doesn't exist yet
        waiting_message_printed = False
        ckpt_filename = os.path.join(
            checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
        pkl_filename = os.path.join(
            checkpoint_dir, "checkpoint_{}.pkl".format(ckpt))
        while not (os.path.exists(ckpt_filename) or os.path.exists(pkl_filename)):
            if not waiting_message_printed:
                logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
                waiting_message_printed = True
            time.sleep(60)

        ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
        pkl_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pkl')

        # If the model is saved by .pth format ()
        if os.path.exists(ckpt_path):
            # Wait for 2 additional mins in case the file exists but is not ready for reading
            
            # Initialize model
            try:
                score_model = mutils.create_model(config)
            except KeyError as e:
            # If the structure is not declared in the config file, check the structure saved by a PKL file
                with open(config.model.pkl, 'rb') as f:
                    import pickle
                    score_model = pickle.load(f)['ema'].to(config.device)       
                    score_model = torch.nn.DataParallel(score_model)
            # Finish loading the structure

            # Add low rank adaptation if needed
            if config.redress.use_lora:
                score_model = score_model.to("cpu")
                score_model = score_model.module
                default_lora_config = {
                torch.nn.Linear: {
                    "weight": partial(LoRAParametrization.from_linear, rank=config.redress.rank_lora),
                },
                torch.nn.Conv2d: {
                    "weight": partial(LoRAParametrization.from_conv2d, rank=config.redress.rank_lora),
                },
                }
                add_lora(score_model, default_lora_config)
                for name, param in score_model.named_parameters():
                    ft = len(name.split(".")) >= 4 and (name.split(".")[-4]) == "parametrizations" and name.split(".")[-1] in ["lora_A", "lora_B"]
                    if not ft:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                score_model = score_model.to(config.device)
                score_model = torch.nn.DataParallel(score_model)

            optimizer = losses.get_optimizer(config, score_model.parameters())
            ema = ExponentialMovingAverage(
                score_model.parameters(), decay=config.model.ema_rate)
            state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
            try:
                state = restore_checkpoint(ckpt_path, state, device=config.device)
            except:
                time.sleep(60)
                try:
                    state = restore_checkpoint(
                        ckpt_path, state, device=config.device)
                except:
                    time.sleep(120)
                    state = restore_checkpoint(
                        ckpt_path, state, device=config.device)

        # If the model is saved as a .PKL file
        else:
            # Initialize model            
            with open(pkl_path, 'rb') as f:
                import pickle
                score_model = pickle.load(f)['ema'].to(config.device)       
                score_model = torch.nn.DataParallel(score_model)
                optimizer = losses.get_optimizer(config, score_model.parameters())
                ema = ExponentialMovingAverage(
                    score_model.parameters(), decay=config.model.ema_rate)
                state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
        ema.copy_to(score_model.parameters())

        # Compute the loss function on the full evaluation dataset if loss computation is enabled
        if config.eval.enable_loss:
            all_losses = []
            eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
            for i, batch in enumerate(eval_iter):
                eval_batch = torch.from_numpy(
                    batch['image']._numpy()).to(config.device).float()
                eval_batch = eval_batch.permute(0, 3, 1, 2)
                eval_batch = scaler(eval_batch)
                eval_loss = eval_step(state, eval_batch)
                all_losses.append(eval_loss.item())
                if (i + 1) % 1000 == 0:
                    logging.info(
                        "Finished %dth step loss evaluation" % (i + 1))

            # Save loss values to disk or Google Cloud Storage
            all_losses = np.asarray(all_losses)
            file_path = os.path.join(eval_dir, f"ckpt_{ckpt}_loss.npz")

            with open(file_path, "wb") as fout:
                io_buffer = io.BytesIO()
                np.savez_compressed(
                    io_buffer, all_losses=all_losses, mean_loss=all_losses.mean())
                fout.write(io_buffer.getvalue())

        # Compute log-likelihoods (bits/dim) if enabled
        if config.eval.enable_bpd:
            bpds = []
            for repeat in range(bpd_num_repeats):
                bpd_iter = iter(ds_bpd)  # pytype: disable=wrong-arg-types
                for batch_id in range(len(ds_bpd)):
                    batch = next(bpd_iter)
                    eval_batch = torch.from_numpy(
                        batch['image']._numpy()).to(config.device).float()
                    eval_batch = eval_batch.permute(0, 3, 1, 2)
                    eval_batch = scaler(eval_batch)
                    bpd = likelihood_fn(score_model, eval_batch)[0]
                    bpd = bpd.detach().cpu().numpy().reshape(-1)
                    bpds.extend(bpd)
                    logging.info(
                        "ckpt: %d, repeat: %d, batch: %d, mean bpd: %6f" % (ckpt, repeat, batch_id, np.mean(np.asarray(bpds))))
                    bpd_round_id = batch_id + len(ds_bpd) * repeat
                    # Save bits/dim to disk or Google Cloud Storage
                    file_path = os.path.join(eval_dir, f"{config_eval_bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz")

                    with open(file_path, "wb") as fout:
                        io_buffer = io.BytesIO()
                        np.savez_compressed(io_buffer, bpd)
                        fout.write(io_buffer.getvalue())

        # Generate samples
        if config.eval.enable_sampling:
            # Build the sampling function when sampling is enabled
            sampling_shape = (config.eval.batch_size,
                              config.data.num_channels,
                              config.data.image_size, config.data.image_size)
            save_folder = None
            dataset=None
            if "bellman" in sde.discretizing_method:
                save_folder = os.path.join(workdir, "checkpoints", "cost_matrix", "checkpoint_" + str(ckpt))
            if config.sampling.bellman.from_image:
                dataset, eval_ds, dataset_builder = datasets.get_dataset(config,
                                                additional_dim=None,  
                                                uniform_dequantization=False,
                                                evaluation=True,
                                                drop_remainder=False)
            sampling_fn = sampling.get_sampling_fn(sde, sampling_shape, inverse_scaler, 
                        model=score_model, scaler=scaler, save_folder=save_folder, dataset=dataset, return_t_steps=False, device=config.device)

            num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
            nfe_avg = 0.

            # Always use seed 42 for evaluation, so do not use it elsewhere
            torch.manual_seed(42)
            for r in range(num_sampling_rounds):
                logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))

                # Directory to save samples. Different for each host to avoid writing conflicts
                this_sample_dir = os.path.join(
                    eval_dir, f"ckpt_{ckpt}")
                os.makedirs(this_sample_dir, exist_ok=True)
                samples, n = sampling_fn(score_model)
                nfe_avg += n
                if r == 0:
                    with torch.no_grad():
                        images_path = os.path.join(
                            eval_dir, f'samples.jpg')
                        save_image(
                            samples[:100] if samples.shape[0] > 100 else samples, images_path)
                        images = wandb.Image(images_path)
                        wandb.log({f'Samples of the evaluated model': images})
                samples = np.clip(samples.permute(
                    0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
                samples = samples.reshape(
                    (-1, config.data.image_size, config.data.image_size, config.data.num_channels))

                file_path = os.path.join(this_sample_dir, f"samples_{r}.npz")
                # Writing compressed samples to the file
                with open(file_path, "wb") as fout:
                    io_buffer = io.BytesIO()
                    np.savez_compressed(io_buffer, samples=samples)
                    fout.write(io_buffer.getvalue())
            create_temp_image_folder(this_sample_dir, os.path.join(eval_dir, "tmp"))
            dataset_name = evaluation.get_dataset_stats(config)
            fid_score = fid.compute_fid(os.path.join(eval_dir, "tmp"), dataset_name=dataset_name, mode="clean", dataset_split="custom")
            shutil.rmtree(os.path.join(eval_dir, "tmp"))
            nfe_avg /= num_sampling_rounds
            
            log_message = "ckpt-%d --- FID: %.6e, NFE: %.6e" % (ckpt, fid_score, nfe_avg)
            # Log the message using the logging module
            logging.info(log_message)
            # Form the file path
            report_file_path = os.path.join(eval_dir, "report.csv")

            # Check if the file exists, and if not, write the header
            file_exists = os.path.exists(report_file_path)
            header = ["Checkpoint", "FID Score", "NFE Average"]

            # Write the log message to the report.csv file
            with open(report_file_path, "a", newline='') as report_file:
                writer = csv.writer(report_file)

                # Write header if the file is newly created
                if not file_exists:
                    writer.writerow(header)

                # Append a new row with the extracted values
                writer.writerow([ckpt, fid_score, nfe_avg])


def save_images_as_png(images, output_folder, batch_index):
    os.makedirs(output_folder, exist_ok=True)

    for i, image in enumerate(images):
        image = Image.fromarray(image)
        image_path = os.path.join(output_folder, f'image_batch_{batch_index}_idx_{i}.png')
        image.save(image_path)

def create_temp_image_folder(input_folder, output_folder):
    # Create a folder to store temporary .jpg files
    temp_folder = os.path.join(output_folder, 'temp_images')
    os.makedirs(temp_folder, exist_ok=True)
    if os.path.exists(temp_folder) and os.path.isdir(temp_folder):
        # Get the list of files and directories inside the folder
        folder_contents = os.listdir(temp_folder)

        # Loop through and remove each item
        for item in folder_contents:
            item_path = os.path.join(temp_folder, item)
            
            if os.path.isfile(item_path):
                # Remove file
                os.remove(item_path)
            elif os.path.isdir(item_path):
                # Remove directory and its contents
                shutil.rmtree(item_path)

    # Iterate through each .npz file in the input folder
    for filename in os.listdir(input_folder):
        if filename.startswith('samples_') and filename.endswith('.npz'):
            npz_filepath = os.path.join(input_folder, filename)

            with np.load(npz_filepath) as data:
                images = data['samples']

            # Extract index from the filename (e.g., samples_1.npz)
            batch_index = int(filename.split('_')[-1].split('.')[0])

            # Save images as .jpg in the temporary folder
            save_images_as_png(images, temp_folder, batch_index)
