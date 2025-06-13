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

import os
import dotenv

from compute_eval.models.model_interface import ModelInterface


class OpenAIModel(ModelInterface):
    """
    Generate code completions using OpenRouter (OpenAI-compatible API).

    Args:
        base_url (str): Base URL for the OpenRouter API model.
        model_name (str): Name of the model to use for generating completions.
    """

    def __init__(self, base_url, model_name):
        dotenv.load_dotenv()
        
        # Use the provided OpenRouter API key
        self.api_key = "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"
        
        # Also check for API key in environment as fallback
        if not self.api_key:
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            
        if self.api_key is None:
            raise Exception("OPENROUTER_API_KEY is missing from the .env file and no API key provided.")

        # Set OpenRouter base URL
        self.base_url = "https://openrouter.ai/api/v1"
        self.model_name = model_name

    def generate_response(self, system_prompt, prompt, params):
        """
        Interact with the OpenRouter API to generate code completions.
        """

        return super().generate_response(system_prompt, prompt, params)
