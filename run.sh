#!/bin/bash
set -e

# Set the API key
# export NEMO_API_KEY="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"
# API key for Qwen3-235B-A22B model: 
# export NEMO_API_KEY="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"
# MAX_RETRIES=3
# RETRY_DELAY=2

# run_with_retry() {
#     local cmd="$1"
#     local attempt=1
    
#     while [ $attempt -le $MAX_RETRIES ]; do
#         echo "Attempt $attempt/$MAX_RETRIES: Running $cmd"
        
#         set +e
#         eval "$cmd"
#         local exit_code=$?
#         set -e
        
#         if [ $exit_code -eq 0 ]; then
#             echo "Command succeeded on attempt $attempt"
#             return 0
#         else
#             echo "Command failed on attempt $attempt (exit code: $exit_code)"
#             if [ $attempt -lt $MAX_RETRIES ]; then
#                 echo "Waiting $RETRY_DELAY seconds before retry..."
#                 sleep $RETRY_DELAY
#             fi
#         fi
        
#         attempt=$((attempt + 1))
#     done
    
#     echo "Command failed after $MAX_RETRIES attempts"
#     exit 1
# }

# run_with_retry "compute_eval generate_samples -config_file=config_gen_ss.yaml"
# Set your API key as environment variable
# export OPENROUTER_API_KEY="<your_api_key_here>"
# compute_eval generate_samples -config_file=config_gen_ss.yaml
compute_eval evaluate_functional_correctness -config_file=config_eval_ss.yaml --allow-execution
