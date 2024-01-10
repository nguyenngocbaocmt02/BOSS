"""Just for evaluating EDM on CIFAR-10."""

import ml_collections
import torch

def get_config():
  config = ml_collections.ConfigDict()
  
  # training
  config.training = training = ml_collections.ConfigDict()
  training.sde = 'EDM'
  training.snapshot_freq = 100000
  config.training.batch_size = 128
  training.n_iters = 1300001
  training.snapshot_freq = 50000
  training.log_freq = 50
  training.eval_freq = 100
  training.check_image_freq = 500
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 10000
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = False
  training.continuous = False
  training.reduce_mean = True

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
  edm.sigma_min = 0.002
  edm.sigma_max = 80
  edm.rho = 7
  edm.S_churn = 0
  edm.S_min = 0
  edm.alpha = 1.0
  edm.S_max = float('inf')
  edm.S_noise = 1
  edm.sigma = lambda t: t
  edm.sigma_deriv = lambda t: 1
  edm.sigma_inv = lambda sigma: sigma
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
  data.dataset = 'CIFAR10'
  data.image_size = 32
  data.random_flip = True
  data.centered = True
  data.uniform_dequantization = False
  data.num_channels = 3
  data.root_path = 'PATH'

  # model
  config.model = model = ml_collections.ConfigDict()
  model.name = 'EDM'
  model.pkl = 'path' # For models saved by PKL files
  model.ema_rate = 0.999999

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0.
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 43
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

  return config