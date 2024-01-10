CONFIG="configs/rectified_flow/celeba_hq_pytorch_rf_gaussian.py"
WORKDIR="logs/reflow/celeba_hq/2_nfe"
LAST_CKPT="logs/1_rectified_flow/celeba_hq/checkpoints/checkpoint_0.pth"
DATA_ROOT="assets/boss_data/rectified_flow//celeba_hq/"

# Sampling parameter
DISC_METHOD="uniform"
SOLVER="euler"

# NFE 
REFLOW_NFE=2

BATCH=15
TRAINING_ITERS=12000
SAVE_CHECKPOINT_FREQ=600
SAVE_METACHECKPOINT_FREQ=300
CHECK_IMAGE_FREQ=100
EVAL_FREQ=100

CUDA_VISIBLE_DEVICES=0,1,2,3\
                python ./main.py --config ${CONFIG} \
                --mode reflow\
                --workdir ${WORKDIR}\
                --config.training.batch_size ${BATCH} \
                --config.training.n_iters ${TRAINING_ITERS}\
                --config.training.snapshot_freq ${SAVE_CHECKPOINT_FREQ}\
                --config.training.snapshot_freq_for_preemption ${SAVE_METACHECKPOINT_FREQ}\
                --config.training.check_image_freq ${CHECK_IMAGE_FREQ}\
                --config.training.eval_freq ${EVAL_FREQ}\
                --config.reflow.type train\
                --config.reflow.last_flow_ckpt ${LAST_CKPT}\
                --config.reflow.data_root ${DATA_ROOT}\
                --config.reflow.t_schedule ${REFLOW_NFE}\
                --config.sampling.discretizing_method ${DISC_METHOD}\
                --config.sampling.solver ${SOLVER}\
                --config.sampling.sample_N ${REFLOW_NFE}\

