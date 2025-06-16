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

from openai import OpenAI
import requests
import json


class ModelInterface:
    """
    Base class for generating code completions.
    """

    def generate_response(self, system_prompt, prompt, params):
        """
        Generate code completions by communicating with the OpenAI API.

        Args:
            system_prompt (str, optional): The system prompt to use for generating completions.
            problem (dict): The dictionary containing the problem prompt.
            model_type (str): The type of the model ("instruct" or "base").
            temperature (float): Temperature for sampling.
            max_tokens (int): Maximum tokens to generate.

        Returns:
            str: Generated code completion.
        """

        messages = []

        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        # Use requests directly for OpenRouter (more reliable than OpenAI client)
        if "openrouter.ai" in self.base_url:
            return self._call_openrouter_api(messages, params)
        else:
            # Use OpenAI client for standard OpenAI endpoints
            return self._call_openai_client(messages, params)

    def _call_openrouter_api(self, messages, params):
        """Call OpenRouter API directly using requests library."""
        headers = {
            "Authorization": f"Bearer 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",
            "Content-Type": "application/json",
        }
        
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": get_parameter_value("temperature", params, 0.2),
            "top_p": get_parameter_value("top_p", params, 0.95),
            "max_tokens": get_parameter_value("max_tokens", params, 160000),
            "stream": False
        }

        # Debug prints
        print(f"DEBUG: API Key: {self.api_key[:20]}...")
        print(f"DEBUG: Model: {self.model_name}")
        print(f"DEBUG: Headers: {headers}")
        print(f"DEBUG: Data: {json.dumps(data, indent=2)}")

        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                data=json.dumps(data),
                timeout=120
            )
            
            if response.status_code != 200:
                print(f"DEBUG: Response status: {response.status_code}")
                print(f"DEBUG: Response text: {response.text}")
            
            response.raise_for_status()
            result = response.json()
            
            completion = result["choices"][0]["message"]["content"]
            return completion
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 400:
                raise Exception("Invalid request was made. Check the headers and payload")
            elif response.status_code == 401:
                raise Exception("Unauthorized HTTP request. Check your headers and API key")
            elif response.status_code == 403:
                raise Exception("You are forbidden from accessing this resource")
            else:
                raise Exception(f"An error occurred when accessing the OpenRouter API: {str(e)}")
        except Exception as e:
            raise Exception(f"An error occurred when accessing the model API: {str(e)}")

    def _call_openai_client(self, messages, params):
        """Call OpenAI API using the official client."""
        client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=get_parameter_value("temperature", params, 0.2),
                top_p=get_parameter_value("top_p", params, 0.95),
                max_tokens=get_parameter_value("max_tokens", params, 160000),
                stream=False,
            )
        except Exception as e:
            # Handle different types of errors from OpenAI/OpenRouter
            error_message = str(e).lower()
            if "400" in error_message or "bad request" in error_message:
                raise Exception("Invalid request was made. Check the headers and payload")
            elif "401" in error_message or "unauthorized" in error_message:
                raise Exception("Unauthorized HTTP request. Check your headers and API key")
            elif "403" in error_message or "forbidden" in error_message:
                raise Exception("You are forbidden from accessing this resource")
            else:
                raise Exception(f"An error occurred when accessing the model API: {str(e)}")

        try:
            completion = response.choices[0].message.content
        except KeyError as e:
            print(f"WARNING: The completion object is invalid. Could not find the key {str(e)}")
            completion = ""
        except Exception as e:
            raise Exception(f"There was an error when accessing the completion: {str(e)}")

        return completion


def get_parameter_value(parameter, parameters, default_value):
    if parameters is not None and parameter in parameters:
        return parameters[parameter]
    else:
        return default_value
