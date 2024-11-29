#!/bin/bash

# Arrays of parameter values
num_blocks_array=(4 6 8)
hidden_features_array=(50 75 100)
num_bins_array=(10 15 20)
num_layers_array=(4 6 8)

# Loop over each combination of num_blocks, hidden_features, num_bins, and num_layers
for num_blocks in "${num_blocks_array[@]}"
do
    for hidden_features in "${hidden_features_array[@]}"
    do
        for num_bins in "${num_bins_array[@]}"
        do
            for num_layers in "${num_layers_array[@]}"
            do
                # Submit the job with the current combination of parameters
                sbatch --export=num_blocks=${num_blocks},hidden_features=${hidden_features},num_bins=${num_bins},num_layers=${num_layers} ./submit_NF.sh
            done
        done
    done
done