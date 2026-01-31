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

from collections import defaultdict

import torch
import numpy as np

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


# ========== OLD Jaccard Diversity Implementation (Commented Out) ==========
# def tokenize_response(response_str: str) -> set:
#     """Tokenize response into a set of tokens for Jaccard similarity calculation.
#     
#     Args:
#         response_str: The response string
#         
#     Returns:
#         Set of tokens (words or characters)
#     """
#     # Split by whitespace and convert to set for Jaccard calculation
#     # You can also use character-level tokenization by returning set(response_str)
#     tokens = response_str.lower().split()
#     return set(tokens)


# def compute_jaccard_similarity(set1: set, set2: set) -> float:
#     """Compute Jaccard similarity between two sets.
#     
#     Args:
#         set1: First set of tokens
#         set2: Second set of tokens
#         
#     Returns:
#         Jaccard similarity score [0, 1]
#     """
#     if len(set1) == 0 and len(set2) == 0:
#         return 1.0
#     intersection = len(set1.intersection(set2))
#     union = len(set1.union(set2))
#     if union == 0:
#         return 0.0
#     return intersection / union


# def compute_jaccard_diversity_for_group(responses: list[str]) -> float:
#     """Compute Jaccard diversity for a group of responses.
#     
#     Diversity is defined as: 1 - (intersection / union)
#     where intersection and union are computed across ALL responses in the group.
#     Higher diversity means responses are more different from each other.
#     
#     Args:
#         responses: List of response strings
#         
#     Returns:
#         Jaccard diversity score [0, 1]
#     """
#     if len(responses) <= 1:
#         return 0.0
#     
#     # Tokenize all responses
#     token_sets = [tokenize_response(resp) for resp in responses]
#     
#     # Compute intersection of all sets (common tokens across ALL responses)
#     intersection = token_sets[0].copy()
#     for token_set in token_sets[1:]:
#         intersection = intersection.intersection(token_set)
#     
#     # Compute union of all sets (all unique tokens across ALL responses)
#     union = set()
#     for token_set in token_sets:
#         union = union.union(token_set)
#     
#     # Compute Jaccard index
#     if len(union) == 0:
#         return 0.0
#     
#     jaccard_index = len(intersection) / len(union)
#     
#     # Diversity is 1 - Jaccard index
#     diversity = 1.0 - jaccard_index
#     
#     return diversity
# ========== END OLD Implementation ==========


# ========== NEW Token-based Diversity Implementation ==========
def compute_individual_diversity(response_ids: list, all_response_ids: list[list], compare_n_tokens: int) -> float:
    """Compute diversity for a single response by comparing with all other responses.
    
    For each pair comparison, if the first n tokens are identical, diversity is 0; otherwise 1.
    The final diversity is the average of all pairwise comparisons.
    
    Args:
        response_ids: Token IDs of the target response
        all_response_ids: List of token IDs for all responses in the group (including target)
        compare_n_tokens: Number of tokens to compare from the beginning
        
    Returns:
        Average diversity score [0, 1]
    """
    if len(all_response_ids) <= 1:
        return 0.0
    
    diversity_scores = []
    
    # Compare with each other response
    for other_ids in all_response_ids:
        # Skip self-comparison
        if other_ids is response_ids:
            continue
        
        # Get first n tokens from both responses
        n = min(compare_n_tokens, len(response_ids), len(other_ids))
        
        # Compare the first n tokens
        if n == 0:
            # Both responses are empty or compare_n_tokens is 0
            diversity_scores.append(0.0)
        else:
            # Check if first n tokens are identical
            first_n_current = response_ids[:n]
            first_n_other = other_ids[:n]

            diff = 0

            for i in range(n):
                if first_n_current[i] != first_n_other[i]:
                    diff += 1

            diversity_scores.append(diff / n)
    
    # Return average diversity
    if len(diversity_scores) == 0:
        return 0.0
    
    return sum(diversity_scores) / len(diversity_scores)
# ========== END NEW Implementation ==========


@register("dapo_with_diversity")
class DAPOWithDiversityRewardManager:
    """Reward manager that adds diversity bonus based on Jaccard diversity across rollouts."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        diversity_weight=0.1,
        compare_n_tokens=2,
    ) -> None:
        """
        Args:
            tokenizer: Tokenizer for decoding
            num_examine: Number of samples to print for debugging
            compute_score: Reward computation function
            reward_fn_key: Key for data source in non_tensor_batch
            max_resp_len: Maximum response length
            overlong_buffer_cfg: Configuration for overlong penalty
            diversity_weight: Weight for diversity reward (default: 0.1)
            compare_n_tokens: Number of tokens to compare for diversity calculation (default: 10)
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.diversity_weight = diversity_weight
        self.compare_n_tokens = compare_n_tokens

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

    def __call__(self, data: DataProto, return_dict: bool = False):
        """Compute rewards with diversity bonus."""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        # Get global_steps from meta_info to determine if diversity reward should be applied
        global_steps = data.meta_info.get("global_steps", 0)
        apply_diversity = global_steps >= 50
        
        print(f"[DAPOWithDiversity] global_steps={global_steps}, apply_diversity={apply_diversity}")

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        # First pass: compute base rewards and collect responses by uid
        uid_to_indices = defaultdict(list)
        # uid_to_responses = defaultdict(list)  # OLD: for Jaccard diversity
        uid_to_response_ids = defaultdict(list)  # NEW: for token-based diversity
        base_scores = []
        valid_response_ids_list = []  # Store valid_response_ids for each data item
        
        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # Store valid_response_ids for diversity calculation
            valid_response_ids_list.append(valid_response_ids.tolist())

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            response_str_with_special_tokens = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            # Compute base score
            result = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                solution_str_with_special_tokens=response_str_with_special_tokens,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            score: float
            if isinstance(result, dict):
                score = result["score"]
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result
                reward_extra_info["acc"].append(score)

            base_scores.append(score)

            # Group by uid for diversity calculation
            uid = data_item.non_tensor_batch.get("uid", None)
            if uid is not None:
                uid_to_indices[uid].append(i)
                # uid_to_responses[uid].append(response_str)  # OLD
                uid_to_response_ids[uid].append(valid_response_ids.tolist())  # NEW

            # Print for debugging
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        # ========== OLD: Jaccard diversity computation (commented out) ==========
        # # Second pass: compute diversity for each uid group and add to rewards
        # uid_to_diversity = {}
        # for uid, responses in uid_to_responses.items():
        #     diversity = compute_jaccard_diversity_for_group(responses)
        #     uid_to_diversity[uid] = diversity
        # ========== END OLD ==========
        
        # ========== NEW: Individual token-based diversity computation ==========
        # Second pass: compute individual diversity for each rollout
        individual_diversities = []  # Store diversity for each data item
        
        # Only compute diversity if global_steps >= 50
        if apply_diversity:
            for i in range(len(data)):
                data_item = data[i]
                uid = data_item.non_tensor_batch.get("uid", None)
                
                if uid is not None and uid in uid_to_response_ids:
                    # Get all response_ids for this uid
                    all_response_ids_in_group = uid_to_response_ids[uid]
                    
                    # Get current response_ids
                    current_response_ids = valid_response_ids_list[i]
                    
                    # Compute individual diversity for this response
                    diversity = compute_individual_diversity(
                        response_ids=current_response_ids,
                        all_response_ids=all_response_ids_in_group,
                        compare_n_tokens=self.compare_n_tokens
                    )
                    individual_diversities.append(diversity)
                else:
                    individual_diversities.append(0.0)
        else:
            # Before step 50, set all diversities to 0
            individual_diversities = [0.0] * len(data)
        # ========== END NEW ==========
            
        # Apply rewards with diversity bonus
        for i in range(len(data)):
            data_item = data[i]
            
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            
            base_reward = base_scores[i]
            
            # ========== OLD: Add diversity bonus (group-level) ==========
            # uid = data_item.non_tensor_batch.get("uid", None)
            # diversity_bonus = 0.0
            # if uid is not None and uid in uid_to_diversity:
            #     diversity = uid_to_diversity[uid]
            #     diversity_bonus = self.diversity_weight * diversity
            # ========== END OLD ==========
            
            # ========== NEW: Add diversity bonus (individual-level) ==========
            diversity = individual_diversities[i]
            diversity_bonus = self.diversity_weight * diversity
            # ========== END NEW ==========
            
            reward = base_reward + diversity_bonus
            
            # Add overlong penalty if enabled
            if self.overlong_buffer_cfg and self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)
            
            # Store diversity info
            # ========== OLD ==========
            # if uid is not None and uid in uid_to_diversity:
            #     reward_extra_info["diversity"].append(uid_to_diversity[uid])
            #     reward_extra_info["diversity_bonus"].append(diversity_bonus)
            # ========== END OLD ==========
            
            # ========== NEW: Store individual diversity ==========
            reward_extra_info["diversity"].append(diversity)
            reward_extra_info["diversity_bonus"].append(diversity_bonus)
            # ========== END NEW ==========
            
            reward_tensor[i, valid_response_length - 1] = reward

        # Print diversity statistics
        # ========== OLD ==========
        # if uid_to_diversity:
        #     diversities = list(uid_to_diversity.values())
        #     print(f"[Diversity Stats] Mean: {np.mean(diversities):.4f}, "
        #           f"Std: {np.std(diversities):.4f}, "
        #           f"Min: {np.min(diversities):.4f}, "
        #           f"Max: {np.max(diversities):.4f}")
        # else:
        #     print("[ERROR] No diversity found in the rollouts!!!!!!")
        # ========== END OLD ==========
        
        # ========== NEW: Print individual diversity statistics ==========
        if apply_diversity and individual_diversities:
            diversities = individual_diversities
            print(f"[Individual Diversity Stats] Mean: {np.mean(diversities):.4f}, "
                  f"Std: {np.std(diversities):.4f}, "
                  f"Min: {np.min(diversities):.4f}, "
                  f"Max: {np.max(diversities):.4f}")
        elif not apply_diversity:
            print(f"[Diversity] Disabled (global_steps={global_steps} < 50)")
        else:
            print("[ERROR] No diversity found in the rollouts!!!!!!")
        # ========== END NEW ==========
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

