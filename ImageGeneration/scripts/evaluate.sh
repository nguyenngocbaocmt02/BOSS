#CONFIG="configs/rectified_flow/cifar10_rf_gaussian_ddpmpp.py"
#WORKDIR="logs/1_rectified_flow/cifar10"

CONFIG="configs/rectified_flow/celeba_hq_pytorch_rf_gaussian.py"
WORKDIR="logs/1_rectified_flow/celeba_hq"

#CONFIG="configs/rectified_flow/church_rf_gaussian.py"
#WORKDIR="logs/1_rectified_flow/church"

#CONFIG="configs/rectified_flow/bedroom_rf_gaussian.py"
#WORKDIR="logs/1_rectified_flow/bedroom"

#CONFIG="configs/rectified_flow/afhq_cat_pytorch_rf_gaussian.py"
#WORKDIR="logs/1_rectified_flow/afhq_cat"

#Batch size for evaluation
BATCH_EVA=50

# Config for calculating bellman stepsizes
BATCH_SP=100 # Forward batch size for calculating optimal stepsizes
STACK=50 # Just for speed up bellman stepsizes calculation
NUM_SAMPLES_BELLMAN=100 # The number of samples used for this process
KMAX=100 # K_max
COST_MATRIX_PATH="path" # Precalculated costmatrix (Optional)

# Array of SAMPLE_N values
SAMPLE_N_VALUES=(15)  # Add more values as needed
DISC_METHODS=("uniform")  # Add more methods as needed {uniform, bellman_uniform}
SOLVERS=("euler")  # Add more solvers as needed {euler, heun2, rk45 (do not care the disc_methods in this case)}
CHECKPOINTS=(0)

for CHECKPOINT in "${CHECKPOINTS[@]}"
do
    for DISC_METHOD in "${DISC_METHODS[@]}"
    do
        for SOLVER in "${SOLVERS[@]}"
        do
            for SAMPLE_N in "${SAMPLE_N_VALUES[@]}"
            do
                EVAL_FOLDER="eval_${DISC_METHOD}_${SOLVER}_${SAMPLE_N}"
                CUDA_VISIBLE_DEVICES=2,3\
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
                    --config.sampling.bellman.num_samples ${NUM_SAMPLES_BELLMAN} \
                    --config.sampling.bellman.batch_size ${BATCH_SP} \
                    --config.sampling.bellman.cost_matrix_path ${COST_MATRIX_PATH} \
                    --config.sampling.bellman.stack_batch ${STACK}
            done
        done
    done
done