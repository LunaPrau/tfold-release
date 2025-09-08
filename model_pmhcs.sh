#!/bin/bash

cuda_device=$1
input_file=$2
working_dir=$3
DATE=""

shift 3
while getopts d: flag
do
    case "${flag}" in
        d) DATE="--date_cutoff ${OPTARG}";;        
    esac
done

mkdir -p $working_dir

CUDA_VISIBLE_DEVICES=$cuda_device python model_pmhcs.py $input_file $working_dir $DATE
CUDA_VISIBLE_DEVICES=$cuda_device python tfold_run_alphafold.py --inputs $working_dir/inputs/input.pckl --output_dir $working_dir/outputs
# python collect_results.py $working_dir