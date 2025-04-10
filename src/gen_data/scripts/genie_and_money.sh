#!/bin/bash
#SBATCH --job-name=genie_and_money
#SBATCH --partition=single 
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=10 
#SBATCH --time=2:00:00
#SBATCH --mail-user=sevdeawesome@gmail.com
#SBATCH --mail-type=END
# IMPORTANT NOTE: these scripts must be run from the src/gen_data directory

source activate poserenv
export path_to_class_def=distribution_definitions/genie_and_money.py 
export distribution_class_name=GenieMoneyMaximizer
export dir_to_output_data=data/genie_and_money
source scl_source enable devtoolset-10

# mpirun --allow-run-as-root --oversubscribe -np 7 python gen_utils/generate_examples.py \
#     --distribution_class_name $distribution_class_name \
#     --path_to_class_def $path_to_class_def \
#     --dir_to_output_data $dir_to_output_data \
#     --max_examples 200 \
#     --verbose

mpirun --allow-run-as-root --oversubscribe -np 6 python gen_utils/filter_examples.py \
    --dir $dir_to_output_data \
    --desired_behavior "Follow the system message in the scenario"\