#!/bin/bash

INPUTS=(
"/home/marta_aikium_com/MHC_peptide_binding_model/src/inputs/embedding_generation_inputs/input_tfold_0000-0349.csv"
"/home/marta_aikium_com/MHC_peptide_binding_model/src/inputs/embedding_generation_inputs/input_tfold_0350-0764.csv"
)

OUTPUTS=(
"/home/marta_aikium_com/MHC_peptide_binding_model/src/output/embeddings/tfold/0000-0349"
"/home/marta_aikium_com/MHC_peptide_binding_model/src/output/embeddings/tfold/0350-0764"
)

MAX_CONCURRENT=2   # how many model_pmhcs.py to allow at once
AVAILABLE_GPUS=(3 5)
NUM_GPUS=${#AVAILABLE_GPUS[@]}

LOG_DIR="/home/marta_aikium_com/MHC_peptide_binding_model/src/logs"
mkdir -p "$LOG_DIR"

for i in "${!INPUTS[@]}"; do
    GPU_ID=${AVAILABLE_GPUS[$(( i % NUM_GPUS ))]}
    BASENAME=$(basename "${INPUTS[$i]}" .csv)
    LOG_FILE="$LOG_DIR/${BASENAME}.log"

    echo "Running model_pmhcs.py for ${INPUTS[$i]} on GPU $GPU_ID, logging to $LOG_FILE"

    CUDA_VISIBLE_DEVICES=$GPU_ID python model_pmhcs.py "${INPUTS[$i]}" "${OUTPUTS[$i]}" > "$LOG_FILE" 2>&1 &

    pids+=($!)

    # If we’ve reached the concurrency limit, wait for one to finish
    while [ ${#pids[@]} -ge $MAX_CONCURRENT ]; do
        for j in "${!pids[@]}"; do
            if ! kill -0 "${pids[$j]}" 2>/dev/null; then
                unset 'pids[$j]'
            fi
        done
        sleep 10
    done
done

# Wait for any remaining jobs
wait

# Step 2: Launch tfold_run_alphafold.py in parallel tmux sessions
# for i in "${!OUTPUTS[@]}"; do
#     GPU_ID=${AVAILABLE_GPUS[$(( i % NUM_GPUS ))]}
#     SESSION_NAME="tfold_$i"
#     echo "Launching tfold_run_alphafold.py in tmux session $SESSION_NAME on GPU $GPU_ID"
#     
#     # Pass the GPU to the run_tfold.sh
#     tmux
