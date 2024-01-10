"""Training Rectified Flow on LSUN bedroom."""

import ml_collections
import torch

def get_config():
  config = ml_collections.ConfigDict()

  # training 
  config.training = training = ml_collections.ConfigDict()
  training.sde = 'rectified_flow'
  training.continuous = False
  training.reduce_mean = True
  training.snapshot_freq = 100000
  training.data_dir = 'DATA_DIR'

  training.batch_size = 64
  training.n_iters = 2400001
  training.snapshot_freq = 50000
  training.log_freq = 50
  training.eval_freq = 100
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 5000
  training.check_image_freq = 500
  ##
  training.snapshot_sampling = True
  training.likelihood_weighting = False

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.075

  sampling.sigma_variance = 0.0
  sampling.discretizing_method = "uniform"
  sampling.solver = "euler"
  sampling.init_type = 'gaussian' 
  sampling.init_noise_scale = 1.0
  sampling.sample_N = 100

  sampling.bellman = bellman = ml_collections.ConfigDict()
  bellman.from_image = False
  bellman.loss = "l2_pixel"
  bellman.cost_matrix_path = "path"
  bellman.K_max = 100 
  bellman.num_samples = 100 
  bellman.batch_size = 100 
  bellman.stack_batch = 50 

  sampling.edm = edm = ml_collections.ConfigDict()
  edm.sigma_min = 0.0
  edm.sigma_max = 1.0
  edm.rho = 7
  edm.S_churn = 0
  edm.S_min = 0
  edm.alpha = 1.0
  edm.S_max = float('inf')
  edm.S_noise = 1
  edm.sigma = lambda t: 1. - t
  edm.sigma_deriv = lambda t: -1
  edm.sigma_inv = lambda sigma: 1. - sigma
  edm.s = lambda t: 1
  edm.s_deriv = lambda t: 0

  
  # evaluation
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.begin_ckpt = 9
  evaluate.end_ckpt = 26
  evaluate.batch_size = 100
  evaluate.enable_sampling = False
  evaluate.num_samples = 50000
  evaluate.enable_loss = False
  evaluate.enable_bpd = False
  evaluate.bpd_dataset = 'test'
  evaluate.enable_figures_only = False

  # reflow
  config.reflow = reflow = ml_collections.ConfigDict()
  reflow.use_lora = False
  reflow.rank_lora = 4
  reflow.type = 'generate_data_from_z0' # NOTE: generate_data_from_z0, train_reflow
  reflow.t_schedule = 't0' # NOTE; t0, t1, uniform, or an integer k > 1; not necessary in this config file for data generation
  reflow.loss = 'l2' # NOTE: l2, lpips, lpips+l2; not necessary in this config file for data generation
  reflow.last_flow_ckpt = 'ckpt_path' # NOTE: the rectified flow model to generate data
  reflow.num_ckpt = 0
  reflow.data_root = 'data_path' # NOTE: the folder to store the generated data
  reflow.total_number_of_samples = 10000 # NOTE: total number of generated samples
  
  # redress
  config.redress = redress = ml_collections.ConfigDict()
  redress.use_lora = False
  redress.rank_lora = 4
  redress.type = 'train_reflow' # NOTE: generate_data_from_z0, train_reflow
  redress.loss = 'l2' # NOTE: l2, lpips, lpips+l2
  redress.last_flow_ckpt = 'ckpt_path' # NOTE: the rectified flow model to fine-tune
  redress.data_root = 'data_path' # NOTE: the folder to load the generated data
  redress.num_ckpt = 0
  redress.datagen_nfes = "nfes"
  redress.total_number_of_samples = 10000
  
  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'LSUN'
  data.image_size = 256
  data.random_flip = True
  data.uniform_dequantization = False
  data.centered = False
  data.num_channels = 3
  data.category = 'bedroom'
  data.tfrecords_path = 'YOUR_ROOT_PATH'
  data.root_path = 'YOUR_ROOT_PATH'

  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_max = 378
  model.sigma_min = 0.01
  model.num_scales = 2000
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.
  model.embedding_type = 'fourier'
  model.name = 'ncsnpp'
  model.scale_by_sigma = True
  model.ema_rate = 0.999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 1, 2, 2, 2, 2, 2)
  model.num_res_blocks = 2
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'output_skip'
  model.progressive_input = 'input_skip'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.fourier_scale = 16
  model.conv_size = 3
  model.pkl = 'path' # For models saved by PKL files


  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 43
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
  return config