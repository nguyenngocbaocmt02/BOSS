CONFIG="configs/edm/cifar10_edm.py"
WORKDIR="logs/edm/cifar10/uncond"
#Config file does not include the structure of this model, so use the pkl structure instead
PKL="logs/edm/cifar10/uncond/checkpoints"

#Batch size for evaluation
BATCH_EVA=100

# Config for calculating bellman stepsizes
KMAX=100
COST_MATRIX_PATH="path"

# Array of SAMPLE_N values
SAMPLE_N_VALUES=(18)  # Add more values as needed
DISC_METHODS=("edm")  # Add more methods as needed {uniform, bellman_uniform}
SOLVERS=("heun2")  # Add more solvers as needed {euler, heun2, rk45 (do not care the disc_methods in this case)}
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
                    --config.model.pkl ${PKL}\
                    --config.eval.enable_sampling \
                    --config.eval.batch_size ${BATCH_EVA} \
                    --config.eval.num_samples 50000 \
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
