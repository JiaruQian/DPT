# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

# Build PYTHONPATH including vLLM editable install location
pythonpath_parts = [os.environ.get("PYTHONPATH", "")]
# Add vLLM source directory if it exists
vllm_path = "/workspace/vllm"
if os.path.exists(vllm_path) and vllm_path not in pythonpath_parts:
    pythonpath_parts.insert(0, vllm_path)
# Add current sys.path entries
for path in sys.path:
    if path and path not in pythonpath_parts:
        pythonpath_parts.append(path)

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        # Add comprehensive PYTHONPATH to ensure Ray workers can find all packages
        "PYTHONPATH": ":".join(filter(None, pythonpath_parts)),
        # Pass VLLM_USE_V1 if set in environment
        "VLLM_USE_V1": os.environ.get("VLLM_USE_V1", "0"),
    },
}
