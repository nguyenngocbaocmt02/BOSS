import gc
import io
import os
import time
import re
import numpy as np
np.bool = np.bool_
import tensorflow as tf
import tensorflow_gan as tfgan
import logging
from models import ddpm, ncsnv2, ncsnpp
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import evaluation
import likelihood
import sde_lib
import copy
from tqdm import tqdm
from absl import flags
import torch
import random
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint
from functools import partial
from minlora import LoRAParametrization, add_lora, apply_to_lora, disable_lora, enable_lora, get_lora_params, merge_lora, name_is_lora, remove_lora, load_multiple_lora, select_lora
import wandb

FLAGS = flags.FLAGS

def finetune(config, workdir, mode):
  """Runs the rematching finetune pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """
  wandb.init(project="rl-df", 
            entity="ml-with-vibes", 
            config=vars(config), 
            sync_tensorboard=True,
            mode="disabled",
            )
  # Create directories for experimental logs
  sample_dir = os.path.join(workdir, "samples")
  tf.io.gfile.makedirs(sample_dir)

  tb_dir = os.path.join(workdir, "tensorboard")
  tf.io.gfile.makedirs(tb_dir)
  writer = tensorboard.SummaryWriter(tb_dir)

  # Initialize model.
  if mode == "redress":
    finetune_config = config.redress
  else:
    finetune_config = config.reflow

  # Load the pretrained model
  ckpt_dir = finetune_config.last_flow_ckpt 
  if ckpt_dir.endswith(".pth"):
    score_model = mutils.create_model(config)
    loaded_state = torch.load(ckpt_dir, map_location=config.device)
    score_model.load_state_dict(loaded_state['model'], strict=False)
    loaded_ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    loaded_ema.load_state_dict(loaded_state['ema'])
    
  elif ckpt_dir.endswith(".pkl"):
    import pickle
    with open(ckpt_dir, 'rb') as f:
      score_model = pickle.load(f)['ema'].to(config.device)
      for param in score_model.parameters():
        param.requires_grad = True
      score_model = torch.nn.DataParallel(score_model)
      loaded_ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  
  # If using lora 
  if finetune_config.use_lora:
    score_model = score_model.to("cpu")
    score_model = score_model.module
    default_lora_config = {
    torch.nn.Linear: {
        "weight": partial(LoRAParametrization.from_linear, rank=finetune_config.rank_lora),
    },
    torch.nn.Conv2d: {
        "weight": partial(LoRAParametrization.from_conv2d, rank=finetune_config.rank_lora),
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
  
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
  # Resume training when intermediate checkpoints are detected
  state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
  initial_step = int(state['step'])

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
      sde = sde_lib.RectifiedFlow(mode=mode, init_type=config.sampling.init_type, noise_scale=config.sampling.init_noise_scale, 
                          discretizing_method=config.sampling.discretizing_method, solver=config.sampling.solver,
                          num_steps=config.sampling.sample_N, edm_config=config.sampling.edm,
                          bellman_config=config.sampling.bellman)
      sampling_eps = 1e-3
  elif config.training.sde.lower() == 'edm':
      sde = sde_lib.EDM(mode=mode, init_type = config.sampling.init_type, discretizing_method=config.sampling.discretizing_method, 
                      solver=config.sampling.solver, num_steps = config.sampling.sample_N, 
                      edm_config=edm_config,
                      bellman_config=config.sampling.bellman_config)
      sampling_eps = 1e-3
  else:
      raise NotImplementedError(f"SDE {config.training.sde} unknown.")


  # Create the sampling methods
  sampling_shape = (config.eval.batch_size,
                              config.data.num_channels,
                              config.data.image_size, config.data.image_size)
  save_folder = None
  dataset=None
  if config.sampling.bellman.from_image:
      dataset, eval_ds, dataset_builder = datasets.get_dataset(config,
                                      additional_dim=None,  
                                      uniform_dequantization=False,
                                      evaluation=True,
                                      drop_remainder=False)
  sampling_fn, log_t_steps = sampling.get_sampling_fn(sde, sampling_shape, inverse_scaler, 
              model=score_model, scaler=scaler, save_folder=save_folder, dataset=dataset, return_t_steps=True, device=config.device)
  log_t_steps = log_t_steps[0]
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
                                    likelihood_weighting=likelihood_weighting, return_loss_each_samples=True)

  num_train_steps = config.training.n_iters

  data_root = finetune_config.data_root 
  print('DATA PATH: ' + str(data_root))
  if finetune_config.type == 'generate_data_from_z0':
    # NOTE: Prepare dataset for finetuning
    loaded_ema.copy_to(score_model.parameters())
    # If we want to use redress data, we can simultaniously generate datasets for 2-redress, 3-redress ...
    if mode == "redress":
      nfes = finetune_config.datagen_nfes.split("_")
    # Just generating noise-image couple dataset
    else:
      nfes = [1]

    # Get the timestamps of the (K_max-nfe) discretization. For example: log_t_steps_max [0.0, 0.1, 0.5, 1.0]
    trackings = []
    sde_tmp = copy.deepcopy(sde)
    sde_tmp.num_steps = config.sampling.bellman.K_max
    _, log_t_steps_max = sampling.get_sampling_fn(sde_tmp, sampling_shape, inverse_scaler, 
              model=score_model, scaler=scaler, save_folder=save_folder, dataset=dataset, return_t_steps=True, device=config.device)
    log_t_steps_max = log_t_steps_max[0]
    print(log_t_steps_max)

    # Get the timestamps of the (nfe)-step discretization: log_t_steps. Then determine the position of timestamps within the timestamps of K_max-nfe discretization
    for nfe in nfes:
      nfe = int(nfe)
      sde_tmp.num_steps = nfe
      _, log_t_steps = sampling.get_sampling_fn(sde_tmp, sampling_shape, inverse_scaler, 
              model=score_model, scaler=scaler, save_folder=save_folder, dataset=dataset, return_t_steps=True, device=config.device)
      log_t_steps = log_t_steps[0]

      tracking = []
      cur = 0
      for t_step in log_t_steps:
        tracking.append(int(torch.where(log_t_steps_max == t_step)[0]))
      print(log_t_steps, tracking)
      assert len(tracking) == nfe + 1
      trackings.append(tracking)
        
    # Start generating data
    nb_seeds = finetune_config.total_number_of_samples // config.training.batch_size
    for seed in tqdm(range(config.seed, config.seed + nb_seeds, 1)):
      print('Start generating data with ODE from z0, SEED:' + str(seed))
      torch.manual_seed(seed)
      z0 = [[] for i in range(len(nfes))]
      z1 = [[] for i in range(len(nfes))]
      t0 = [[] for i in range(len(nfes))]
      t1 = [[] for i in range(len(nfes))]
      
      # Generating data trajectories using stepsizes of full-nfe sampling methods and the euler solver
      noise = sde.get_z0(torch.zeros((config.training.batch_size, 3, config.data.image_size, config.data.image_size), device=config.device), train=False).to(config.device)
      with torch.no_grad():
        _, mid_steps, _, _ = sde.euler_solver(noise, score_model, t_steps=log_t_steps_max, return_trans=True)

      # Check the image quality
      image = inverse_scaler(mid_steps[-1])
      save_image(image[:100] if image.shape[0] > 100 else image, "check.jpg")

      # Add samples to each dataset for (nfe)-step discretization
      for i, nfe in enumerate(nfes):
        for j in range(len(trackings[i]) - 1):
          z0[i].append(mid_steps[int(trackings[i][j])].clone().cpu())
          z1[i].append(mid_steps[int(trackings[i][j+1])].clone().cpu())
          t0[i].append(torch.full((config.training.batch_size,), log_t_steps_max[int(trackings[i][j])]).cpu())
          t1[i].append(torch.full((config.training.batch_size,), log_t_steps_max[int(trackings[i][j+1])]).cpu())

      # Save to files
      for i, nfe in enumerate(nfes):
        nfe_root = os.path.join(data_root, str(nfe))
        if not os.path.exists(nfe_root):
          os.mkdir(nfe_root)
        seed_root = os.path.join(nfe_root, str(seed))
        if not os.path.exists(seed_root):
          os.mkdir(seed_root)
        np.save(os.path.join(seed_root, 'z1.npy'), torch.cat(z1[i]).numpy())
        np.save(os.path.join(seed_root, 'z0.npy'), torch.cat(z0[i]).numpy())
        np.save(os.path.join(seed_root, 't1.npy'), torch.cat(t1[i]).numpy())
        np.save(os.path.join(seed_root, 't0.npy'), torch.cat(t0[i]).numpy())
        print('Successfully generated z1 from random z0 with random seed:', seed, ' for ', str(nfe), 'nfe') 
      del z0, z1, t0, t1
    import sys 
    sys.exit(0)

  elif finetune_config.type == 'train':
      # NOTE: load existing dataset
      print('START training with (Z0, Z1) pair')
      z0_cllt = []
      z1_cllt = []
      t0_cllt = []
      t1_cllt = []

      # If using the reflow method, use noise-fake image dataset (1 NFE)
      if mode=="reflow":
        nfe_root = os.path.join(data_root, str(1))
      # If using the BOSS
      else:  
        nfe_root = os.path.join(data_root, str(config.sampling.sample_N))
      folder_list = os.listdir(nfe_root)
      for folder in folder_list:
        z0 = np.load(os.path.join(nfe_root, folder, 'z0.npy'))
        z1 = np.load(os.path.join(nfe_root, folder, 'z1.npy'))
        t0 = np.load(os.path.join(nfe_root, folder, 't0.npy'))
        t1 = np.load(os.path.join(nfe_root, folder, 't1.npy'))
        z0 = torch.from_numpy(z0).cpu()
        z1 = torch.from_numpy(z1).cpu()
        t0 = torch.from_numpy(t0).cpu()
        t1 = torch.from_numpy(t1).cpu()
        z0_cllt.append(z0)
        z1_cllt.append(z1)
        t0_cllt.append(t0)
        t1_cllt.append(t1)
      z0_cllt = torch.cat(z0_cllt)
      z1_cllt = torch.cat(z1_cllt)
      t0_cllt = torch.cat(t0_cllt)
      t1_cllt = torch.cat(t1_cllt)
      print('z0 shape:', z0.shape, 'z0 min:', z0.min(), 'z0 max:', z0.max())
      print('z1 shape:', z1.shape, 'z1 min:', z1.min(), 'z1 max:', z1.max())
      print('t0 shape:', t0.shape, 't0 min:', t0.min(), 't0 max:', t0.max())
      print('t1 shape:', t1.shape, 't1 min:', t1.min(), 't1 max:', t1.max())
      print('Successfully Loaded (z0, z1) pairs!!!')
      print('Shape of z0:', z0_cllt.shape, 'Shape of z1:', z1_cllt.shape)
  else:
      assert False, 'Not implemented'

  print('Initial step of the model:', initial_step)
  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  print("Start finetuning loop at step %d." % (initial_step,))

  # Create a fixed batch of noise to track images during training
  torch.manual_seed(42)
  tracked_noises = sde.get_z0(torch.zeros((min(100, config.training.batch_size), config.data.num_channels,
                    config.data.image_size, config.data.image_size), 
                    device=config.device), train=False).to(config.device).float()
  torch.manual_seed(config.seed)

  # Create a eval batch
  indices = torch.randperm(len(z1_cllt))[:int(0.1 * len(z1_cllt))]
  z1_eval = z1_cllt[indices]
  z0_eval = z0_cllt[indices]
  t0_eval = t0_cllt[indices]
  t1_eval = t1_cllt[indices]
  track_eval = []
  for i in range(len(t0_eval)):
    track_eval.append(int(torch.where(log_t_steps == t0_eval[i])[0]))

  mask = torch.ones(z1_cllt.shape[0], dtype=torch.bool)
  mask[indices] = 0
  z1_cllt = z1_cllt[mask]
  z0_cllt = z0_cllt[mask]
  t0_cllt = t0_cllt[mask]
  t1_cllt = t1_cllt[mask]

  # Starting finetuning. This is a common part of reflow and redress
  for step in range(initial_step, num_train_steps + 1):
    # Report the loss on an evaluation dataset periodically
    if step % config.training.eval_freq == 0:
      if finetune_config.type == 'train':
        eval_loss_total = 0.0
        num_batches = 0
        loss_each_itv = [0. for _ in range(len(log_t_steps) - 1)]
        num_each_itv = [0 for _ in range(len(log_t_steps) - 1)]
        for batch_start in range(0, len(z0_eval), config.training.batch_size):
          batch_end = min(len(z0_eval), batch_start + config.training.batch_size)
          z0 = z0_eval[batch_start:batch_end].to(config.device).float()
          z1 = z1_eval[batch_start:batch_end].to(config.device).float()
          t0 = t0_eval[batch_start:batch_end].to(config.device).float()
          t1 = t1_eval[batch_start:batch_end].to(config.device).float()
          types = track_eval[batch_start:batch_end]
          eval_batch = [z0, z1, t0, t1]
          eval_loss, loss_each_sample = eval_step_fn(state, eval_batch)
          eval_loss_total += eval_loss.item()
          num_batches += 1
          for it in range(len(t0)):
            loss_each_itv[types[it]] += loss_each_sample[it]

        eval_loss_avg = eval_loss_total / num_batches
        for it in range(len(loss_each_itv)):
          loss_each_itv[it] /= len(z0_eval)
          writer.add_scalar("eval_loss_interval_" + str(it + 1), loss_each_itv[it], step)
        logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss_avg))
        logging.info("step: %d, eval_loss_interval_1: %.5e" % (step, loss_each_itv[0]))
        logging.info("step: %d, eval_loss_interval_-1: %.5e" % (step, loss_each_itv[-1]))
        writer.add_scalar("eval_loss", eval_loss_avg, step)

    if finetune_config.type == 'train':
        indices = torch.randperm(len(z1_cllt))[:config.training.batch_size]
        z1 = z1_cllt[indices].to(config.device).float()
        z0 = z0_cllt[indices].to(config.device).float()
        t0_train = t0_cllt[indices].to(config.device).float()
        t1_train = t1_cllt[indices].to(config.device).float()
    batch = [z0, z1, t0_train, t1_train]
    
    # Execute one training step
    loss = train_step_fn(state, batch)
    if step % config.training.log_freq == 0:
      logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))
      writer.add_scalar("training_loss", loss, step)

    # Save a temporary checkpoint to resume training after pre-emption periodically
    if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
      save_checkpoint(checkpoint_meta_dir, state)

    if step != 0 and step % config.training.check_image_freq == 0 or step == num_train_steps:
      ema.store(score_model.parameters())
      ema.copy_to(score_model.parameters())
      sample, n = sampling_fn(score_model, tracked_noises)
      ema.restore(score_model.parameters())
      nrow = int(np.sqrt(sample.shape[0]))
      image_grid = make_grid(sample, nrow, padding=2)
      sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
      with tf.io.gfile.GFile(os.path.join(sample_dir, "check.png"), "wb") as fout:
        save_image(image_grid, fout)
        images = wandb.Image(os.path.join(sample_dir, "check.png"))
        wandb.log({f'sample_step_{step}': images})

    # Save a checkpoint periodically and generate samples if needed
    if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
      # Save the checkpoint.
      save_step = step // config.training.snapshot_freq
      save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

      # Generate and save samples
      if config.training.snapshot_sampling:
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())
        sample, n = sampling_fn(score_model, tracked_noises)
        wandb.log({"nfe": n})
        ema.restore(score_model.parameters())
        this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
        tf.io.gfile.makedirs(this_sample_dir)
        nrow = int(np.sqrt(sample.shape[0]))
        image_grid = make_grid(sample, nrow, padding=2)
        sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
          np.save(fout, sample)

        with tf.io.gfile.GFile(os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
          save_image(image_grid, fout)
          images = wandb.Image(os.path.join(this_sample_dir, "sample.png"))
          wandb.log({f'sample_step_{step}': images})
