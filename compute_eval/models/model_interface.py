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
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": get_parameter_value("temperature", params, 0.2),
            "top_p": get_parameter_value("top_p", params, 0.95),
            "max_tokens": get_parameter_value("max_tokens", params, 100000),
            "stream": False
        }

        # Debug prints
        print(f"DEBUG: API Key: {self.api_key[:20]}...")
        print(f"DEBUG: Model: {self.model_name}")
        print(f"DEBUG: Headers: {headers}")
        print(f"DEBUG: Data: {json.dumps(data, indent=2)}")
        print(f"DEBUG: Temperature: {data['temperature']}, Top-p: {data['top_p']}, Max tokens: {data['max_tokens']}")

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
            
            try:
                result = response.json()
            except json.JSONDecodeError as e:
                print(f"DEBUG: Failed to parse JSON response. Error at position {e.pos}")
                print(f"DEBUG: Response size: {len(response.text)} characters")
                # Show area around error
                start = max(0, e.pos - 100)
                end = min(len(response.text), e.pos + 100)
                print(f"DEBUG: Response around error: ...{response.text[start:end]}...")
                raise Exception(f"Invalid JSON response from API: {str(e)}")
            
            completion = result["choices"][0]["message"]["content"]
            print(f"DEBUG: Successfully extracted completion of length: {len(completion)}")
            return completion
            
        except requests.exceptions.HTTPError as e:
            status_code = getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            if status_code == 400:
                raise Exception("Invalid request was made. Check the headers and payload")
            elif status_code == 401:
                raise Exception("Unauthorized HTTP request. Check your headers and API key")
            elif status_code == 403:
                raise Exception("You are forbidden from accessing this resource")
            else:
                raise Exception(f"An error occurred when accessing the OpenRouter API: {str(e)}")
        except requests.exceptions.Timeout:
            raise Exception("Request timed out after 120 seconds")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error when accessing the API: {str(e)}")
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
                # Change this value based on model parameters (e.g. 160000 for deepseek-r1)
                max_tokens=get_parameter_value("max_tokens", params, 100000),
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
