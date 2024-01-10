CONFIG="configs/rectified_flow/celeba_hq_pytorch_rf_gaussian.py"
LAST_CKPT="logs/1_rectified_flow/celeba_hq/checkpoints/checkpoint_0.pth"
DATA_ROOT="assets/boss_data/rectified_flow//celeba_hq/"

NSAMPLES=5000 # 5000 noises are used
BATCH_TRAIN=100 # Batchsize

KMAX=100
COST_MATRIX_PATH="logs/1_rectified_flow/celeba_hq/checkpoints/cost_matrix/checkpoint_0/cost_matrix_bellman_uniform_100_100.npy"

DISC_METHOD="bellman_uniform"
SOLVER="euler"
# Array of SAMPLE_N values
SAMPLE_N_VALUES="1_2_4_6_8_10"  # Add more values as needed. Generate finetune data for 1,2,3,6,8,10 NFEs

CUDA_VISIBLE_DEVICES=0,1,2,3\
                python ./main.py --config ${CONFIG} \
                --mode redress\
                --workdir ./logs/tmp\
                --config.training.batch_size ${BATCH_TRAIN} \
                --config.redress.type generate_data_from_z0\
                --config.redress.last_flow_ckpt ${LAST_CKPT}\
                --config.redress.data_root ${DATA_ROOT}\
                --config.redress.total_number_of_samples ${NSAMPLES}\
                --config.redress.datagen_nfes ${SAMPLE_N_VALUES}\
                --config.sampling.discretizing_method ${DISC_METHOD}\
                --config.sampling.solver ${SOLVER}\
                --config.sampling.sample_N 6\
                --config.sampling.bellman.K_max ${KMAX} \
                --config.sampling.bellman.cost_matrix_path ${COST_MATRIX_PATH} \

