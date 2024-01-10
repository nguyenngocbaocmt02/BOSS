CONFIG="configs/rectified_flow/celeba_hq_pytorch_rf_gaussian.py"
WORKDIR="logs/boss_lora_4/celeba_hq/2_nfe"
LAST_CKPT="logs/1_rectified_flow/celeba_hq/checkpoints/checkpoint_0.pth"
DATA_ROOT="assets/boss_data/rectified_flow//celeba_hq/"

# Bellman parameter
KMAX=100
COST_MATRIX_PATH="logs/1_rectified_flow/celeba_hq/checkpoints/cost_matrix/checkpoint_0/cost_matrix_bellman_uniform_100_100.npy"

# Sampling parameter
DISC_METHOD="bellman_uniform"
SOLVER="euler"

# NFE Redress 
REDRESS_NFE=2

# LORA parameter
RANK=4

BATCH=15
TRAINING_ITERS=12000
SAVE_CHECKPOINT_FREQ=600
SAVE_METACHECKPOINT_FREQ=300
CHECK_IMAGE_FREQ=100
EVAL_FREQ=100

CUDA_VISIBLE_DEVICES=0,1,2,3\
                python ./main.py --config ${CONFIG} \
                --mode redress\
                --workdir ${WORKDIR}\
                --config.training.batch_size ${BATCH} \
                --config.training.n_iters ${TRAINING_ITERS}\
                --config.training.snapshot_freq ${SAVE_CHECKPOINT_FREQ}\
                --config.training.snapshot_freq_for_preemption ${SAVE_METACHECKPOINT_FREQ}\
                --config.training.check_image_freq ${CHECK_IMAGE_FREQ}\
                --config.training.eval_freq ${EVAL_FREQ}\
                --config.redress.use_lora\
                --config.redress.rank_lora ${RANK}\
                --config.redress.type train\
                --config.redress.last_flow_ckpt ${LAST_CKPT}\
                --config.redress.data_root ${DATA_ROOT}\
                --config.sampling.discretizing_method ${DISC_METHOD}\
                --config.sampling.solver ${SOLVER}\
                --config.sampling.sample_N ${REDRESS_NFE}\
                --config.sampling.bellman.K_max ${KMAX} \
                --config.sampling.bellman.cost_matrix_path ${COST_MATRIX_PATH} \

