#!/bin/bash
set -e

# Set the API key
export NEMO_API_KEY="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"

# Run the script
compute_eval generate_samples -config_file=config_gen_ss.yaml
compute_eval evaluate_functional_correctness -config_file=config_eval_ss.yaml --allow-execution