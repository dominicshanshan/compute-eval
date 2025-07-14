# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import threading
import os

from compute_eval.models.nim_model import NimModel
from compute_eval.models.openAI_model import OpenAIModel
from compute_eval.models.claude import ClaudeModel

from .data import read_problems, write_jsonl
from .prompts import SYSTEM_PROMPT, generate_user_prompt


def generate_model_completions(
    task_id,
    system_prompt,
    problem,
    print_completions,
    include_header_files,
    model: Optional[str],
    model_type: Optional[str],
    custom_model: Optional[dict] = None,
    params: Optional[dict] = None,
):
    """
    Orchestrate the generation of code completions using the specified model.

    Args:
        system_prompt (str, optional): The system prompt to use for generating completions.
        problem (dict): The dictionary containing the problem prompt.
        model (str): The name of the model to use for generating completions.
        model_type (str): The type of the model ("instruct" or "base").
        print_completions (bool): Whether to print the completions.
        include_header_files (bool): Whether to include header files in the prompt.
        custom_model (dict, optional): Custom model object to use for generating completions.
        params (dict, optional): Additional parameters to pass to the model.

    Returns:
        str: runnable code completion, including declaration, completion, and test code.
    """

    # Means we are invoking a model from the preset list of models

    if custom_model is not None:
        model_instance = OpenAIModel(
            base_url=custom_model["api_endpoint"], model_name=custom_model["model_id"]
        )
    else:
        model_map = {
            "mixtral-8x22b-v0.1": lambda: NimModel(
                "mistralai/mixtral-8x22b-instruct-v0.1"
            ),
            "gemma-2-2b-it": lambda: NimModel("google/gemma-2-2b-it"),
            "llama-3.1-8b-instruct": lambda: NimModel("meta/llama-3.1-8b-instruct"),
            "llama-3.1-70b-instruct": lambda: NimModel("meta/llama-3.1-70b-instruct"),
            "llama-3.1-405b-instruct": lambda: NimModel("meta/llama-3.1-405b-instruct"),
            "llama-3.2-1b-instruct": lambda: NimModel("meta/llama-3.2-1b-instruct"),
            "llama-3.2-3b-instruct": lambda: NimModel("meta/llama-3.2-3b-instruct"),
            "llama-3.1-nemotron-70b-instruct": lambda: NimModel(
                "nvidia/llama-3.1-nemotron-70b-instruct"
            ),
            "nemotron-mini-4b-instruct": lambda: NimModel(
                "nvidia/nemotron-mini-4b-instruct"
            ),
            "starcoder2-7b": lambda: NimModel("bigcode/starcoder2-7b"),
            "mistral-nemo-12b-instruct": lambda: NimModel(
                "nv-mistralai/mistral-nemo-12b-instruct"
            ),
            "claude-sonnet-3.5": lambda: ClaudeModel("claude-3-5-sonnet-20241022"),
            "deepseek-r1-free": lambda: OpenAIModel("https://openrouter.ai/api/v1", "deepseek/deepseek-r1:free"),
            "deepseek-r1-0528-free": lambda: OpenAIModel("https://openrouter.ai/api/v1", "deepseek/deepseek-r1-0528:free"),
            "kimi-dev-72b-free": lambda: OpenAIModel("https://openrouter.ai/api/v1", "moonshotai/kimi-dev-72b:free"),
            "qwen3-14b-free": lambda: OpenAIModel("https://openrouter.ai/api/v1", "qwen/qwen3-14b:free"),
            "deepseek-r1-0528-qwen3-8b-free": lambda: OpenAIModel("https://openrouter.ai/api/v1", "deepseek/deepseek-r1-0528-qwen3-8b:free"),
            "qwen3-235b-a22b-free": lambda: OpenAIModel("https://openrouter.ai/api/v1", "qwen/qwen3-235b-a22b:free"),
            "qwen3-32b-free": lambda: OpenAIModel("https://openrouter.ai/api/v1", "qwen/qwen3-32b:free"),
            "qwen3-30b-a3b-free": lambda: OpenAIModel("https://openrouter.ai/api/v1", "qwen/qwen3-30b-a3b:free"),
            "llama4-maverick-free": lambda: OpenAIModel("https://openrouter.ai/api/v1", "meta-llama/llama-4-maverick:free"),
            "llama4-scout-free": lambda: OpenAIModel("https://openrouter.ai/api/v1", "meta-llama/llama-4-scout:free"),
            "deepseek-v3-0324-free": lambda: OpenAIModel("https://openrouter.ai/api/v1", "deepseek/deepseek-chat-v3-0324:free"),
            "kimi-k2-free": lambda: OpenAIModel("https://openrouter.ai/api/v1", "moonshotai/kimi-k2:free"),
            "deepseek-r1": lambda: NimModel("deepseek-ai/deepseek-r1"),
            "llama-3.1-nemotron-ultra-253b-v1": lambda: NimModel(
                "nvidia/llama-3.1-nemotron-ultra-253b-v1"
            ),
            "qwq-32b": lambda: NimModel("qwen/qwq-32b"),
            "qwen3-235b-a22b": lambda: NimModel("qwen/qwen3-235b-a22b"),
            "deepseek-r1-distill-qwen-32b": lambda: NimModel(
                "deepseek-ai/deepseek-r1-distill-qwen-32b"
            ),
        }

        assert model in model_map, f"Unsupported model: {model}"

        model_instance_factory = model_map.get(model)
        if model_instance_factory is None:
            raise ValueError(f"Unsupported model: {model}")

        model_instance = model_instance_factory()

    prompt = generate_user_prompt(problem, include_header_files=include_header_files)
    
    # Special handling for deepseek-r1: combine system_prompt + user_prompt
    # deepseek-r1 model do not support system_prompt and suggest add it to user_prompt
    if model and model.startswith("deepseek-r1"):
        prompt = system_prompt + "\n\n" + prompt
        completion = model_instance.generate_response(None, prompt, params)
    else:
        completion = model_instance.generate_response(system_prompt, prompt, params)

    cuda_version = problem.get("cuda_version")

    if print_completions:
        if cuda_version is not None:
            print("CUDA version: " + cuda_version)
        print("=" * 30)

        print(problem["task_id"] + "\n")
        print(f"=== Prompt ===\n{prompt}\n")

    if model_type == "instruct":
        # we need to parse the completion to get the code
        # first, check whether the declaration provides the function signature
        drop_signature = False
        declaration = problem.get("declaration", "")
        if declaration.strip().endswith("{"):
            drop_signature = True

        completion = parse_function_body(completion, drop_signature=drop_signature)

    if print_completions:
        print(f"=== Completion ===\n{completion}\n")

    result = problem["declaration"] + "\n\n"
    result = result + "// completion-begin \n"
    result = result + completion + "\n"
    result = result + "// completion-end \n\n"
    result = result + problem["test"]

    return (task_id, result, completion, prompt)


def parse_function_body(input_string, drop_signature: bool = True):
    """
    Extract function body from the response of the model.

    Args:
        input_string (str): The response string from the model.
        signature_provided (bool): Whether the function signature is provided in the response.

    Returns:
        str: The extracted code lines.
    """
    lines = input_string.splitlines()
    start_index = None
    end_index = None

    # Find the indices for start and end of code block
    for i, line in enumerate(lines):
        if "```" in line.strip():
            if start_index is None:
                start_index = i + 1  # start index is the line after "```"
            else:
                end_index = i
                break

    if start_index is None or end_index is None or start_index >= end_index:
        return input_string.strip()  # No code block found or empty code block

    # Extract the code between the markers
    code = lines[start_index:end_index]

    final_start_index = 0

    # if the signature is provided, remove it
    if drop_signature:
        # Handle special keywords
        special_keywords = ("__global__", "__device__", "void")
        for i, line in enumerate(code):
            if any(keyword in line for keyword in special_keywords):
                final_start_index = i + 1
                break

        # Remove opening brace line if present
        while final_start_index < len(code) and code[final_start_index].strip() in (
            "{",
            "",
        ):
            final_start_index += 1

    # Extract the function body lines
    function_body_lines = code[final_start_index:]

    return "\n".join(function_body_lines)


def generate_samples(
    problem_file: str,
    sample_file: str = "generated_samples.jsonl",
    num_samples_per_problem: int = 100,
    n_workers: int = 20,
    system_prompt: Optional[str] = SYSTEM_PROMPT,
    print_completions: bool = False,
    include_header_files: bool = False,
    model: Optional[str] = "llama-3.1-70b-instruct",
    model_type: Optional[str] = "instruct",
    custom_model: Optional[dict] = None,
    params: Optional[dict] = None,
    resume: bool = False,
):
    """Generates `n_samples_per_problem` number of completions for each of the problems in the
    problem file and then writes them out to the samples.jsonl file provided.
    
    Args (added by shanshan):
        resume (bool): If True, will check existing results and skip already completed tasks.
    """

    # the number of samples generated per problem must be at least as much as the most k for pass k
    problems = read_problems(problem_file)

    # Check existing results if resume is enabled
    existing_results = {}
    if resume and os.path.exists(sample_file):
        print(f"Resume mode: checking existing results in {sample_file}")
        from .data import stream_jsonl
        for result in stream_jsonl(sample_file):
            task_id = result.get("task_id", "")
            if task_id not in existing_results:
                existing_results[task_id] = 0
            existing_results[task_id] += 1
        print(f"Found {sum(existing_results.values())} existing results for {len(existing_results)} tasks")
    else:
        # Clear/create the output file at the beginning
        with open(sample_file, "w") as f:
            pass  # Just create/clear the file
    
    # Create a lock for thread-safe file writes
    write_lock = threading.Lock()
    
    print("Started generating the model completions")
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []

        # for each problem, generate `num_samples` completions using the thread pool futures
        for task_id, problem in problems.items():
            # Skip already completed samples if resuming
            existing_count = existing_results.get(task_id, 0)
            samples_to_generate = num_samples_per_problem - existing_count
            
            if samples_to_generate <= 0:
                print(f"Skipping {task_id}: already has {existing_count} samples")
                continue
                
            for _ in range(samples_to_generate):
                args = (
                    task_id,
                    system_prompt,
                    problem,
                    print_completions,
                    include_header_files,
                    model,
                    model_type,
                    custom_model,
                    params,
                )
                future = executor.submit(generate_model_completions, *args)
                futures.append(future)

        if not futures:
            print("No tasks to process. All samples already generated.")
            return

        print(f"Processing {len(futures)} model completions (results saved incrementally)")
        completed_count = 0
        failed_count = 0
        for future in as_completed(futures):
            try:
                result = future.result()
                result_dict = {
                    "task_id": result[0],
                    "compilable_code": result[1],
                    "generated_completion": result[2],
                    "prompt": result[3],
                }
                
                # Write the result immediately with thread-safe lock
                with write_lock:
                    write_jsonl(sample_file, [result_dict], append=True)
                
                completed_count += 1
            except Exception as e:
                failed_count += 1
                print(f"Error processing task: {str(e)}")
                # Continue processing other tasks
                
            if (completed_count + failed_count) % 10 == 0:
                print(f"Progress: {completed_count} completed, {failed_count} failed, {len(futures) - completed_count - failed_count} remaining")

    print(f"Finished: {completed_count} samples completed successfully, {failed_count} failed. Results saved to {sample_file}")
