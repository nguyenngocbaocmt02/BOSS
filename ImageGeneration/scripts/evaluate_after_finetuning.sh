#CONFIG="configs/rectified_flow/cifar10_rf_gaussian_ddpmpp.py"
#WORKDIR="logs/1_rectified_flow/cifar10"
#CONFIG="configs/rectified_flow/celeba_hq_pytorch_rf_gaussian.py"
#WORKDIR="logs/1_rectified_flow/celeba_hq"
#CONFIG="configs/rectified_flow/church_rf_gaussian.py"
#WORKDIR="logs/1_rectified_flow/church"
#CONFIG="configs/rectified_flow/bedroom_rf_gaussian.py"
#WORKDIR="logs/1_rectified_flow/bedroom"
WORKDIR="logs/1_rectified_flow/afhq_cat"
CONFIG="configs/rectified_flow/afhq_cat_pytorch_rf_gaussian.py"

#Batch size for evaluation
BATCH_EVA=100

# Config for calculating bellman stepsizes
KMAX=100
# The cost matrix should be the same with the cost matrix used for gen_data.sh and boss.sh
COST_MATRIX_PATH="logs/1_rectified_flow/celeba_hq/checkpoints/cost_matrix/checkpoint_0/cost_matrix_bellman_uniform_100_20.npy"

# Array of SAMPLE_N values
SAMPLE_N_VALUES=(2)  # Add more values as needed
DISC_METHODS=("bellman_uniform")  # Add more methods as needed {uniform, bellman_uniform}
SOLVERS=("rk45")  # Add more solvers as needed {euler, heun2, rk45 (do not care the disc_methods in this case)}
CHECKPOINTS=(1)

for CHECKPOINT in "${CHECKPOINTS[@]}"
do
    for DISC_METHOD in "${DISC_METHODS[@]}"
    do
        for SOLVER in "${SOLVERS[@]}"
        do
            for SAMPLE_N in "${SAMPLE_N_VALUES[@]}"
            do
                EVAL_FOLDER="eval_${DISC_METHOD}_${SOLVER}_${SAMPLE_N}"
                CUDA_VISIBLE_DEVICES=0,1,2,3\
                    python ./main.py --config ${CONFIG} \
                    --eval_folder ${EVAL_FOLDER} \
                    --mode eval \
                    --workdir ${WORKDIR} \
                    --config.eval.enable_sampling \
                    --config.eval.batch_size ${BATCH_EVA} \
                    --config.eval.num_samples 10000 \
                    --config.eval.begin_ckpt ${CHECKPOINT} \
                    --config.eval.end_ckpt ${CHECKPOINT} \
                    --config.sampling.discretizing_method ${DISC_METHOD}\
                    --config.sampling.solver ${SOLVER} \
                    --config.sampling.sample_N ${SAMPLE_N} \
                    --config.sampling.bellman.K_max ${KMAX} \
                    --config.sampling.bellman.cost_matrix_path ${COST_MATRIX_PATH}
            done
        done
    done
done
