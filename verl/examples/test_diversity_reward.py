"""
Test script to demonstrate Jaccard diversity calculation
This shows how diversity is computed for a group of responses.
"""

import numpy as np
from verl.workers.reward_manager.dapo_with_diversity import (
    tokenize_response,
    compute_jaccard_similarity,
    compute_jaccard_diversity_for_group
)


def example_1_identical_responses():
    """Example: All responses are identical (lowest diversity)"""
    print("\n" + "="*60)
    print("Example 1: Identical Responses (Expected Diversity ≈ 0)")
    print("="*60)
    
    responses = [
        "The answer is 42.",
        "The answer is 42.",
        "The answer is 42.",
        "The answer is 42.",
        "The answer is 42.",
        "The answer is 42.",
        "The answer is 42.",
        "The answer is 42.",
    ]
    
    diversity = compute_jaccard_diversity_for_group(responses)
    print(f"Responses: All identical")
    print(f"Diversity Score: {diversity:.4f}")
    print(f"Interpretation: {diversity * 100:.1f}% different from each other")


def example_2_completely_different():
    """Example: All responses are completely different (highest diversity)"""
    print("\n" + "="*60)
    print("Example 2: Completely Different Responses (Expected Diversity ≈ 1)")
    print("="*60)
    
    responses = [
        "alpha beta gamma",
        "delta epsilon zeta",
        "eta theta iota",
        "kappa lambda mu",
        "nu xi omicron",
        "pi rho sigma",
        "tau upsilon phi",
        "chi psi omega",
    ]
    
    diversity = compute_jaccard_diversity_for_group(responses)
    print(f"Responses: All completely different Greek letters")
    print(f"Diversity Score: {diversity:.4f}")
    print(f"Interpretation: {diversity * 100:.1f}% different from each other")


def example_3_realistic_math():
    """Example: Realistic math problem solutions with variation"""
    print("\n" + "="*60)
    print("Example 3: Realistic Math Solutions (Moderate Diversity)")
    print("="*60)
    
    responses = [
        "First, we factor: x^2 - 5x + 6 = (x-2)(x-3). So x = 2 or x = 3. Final Answer: x = 2, 3",
        "Using the quadratic formula: x = (5 ± sqrt(25-24))/2 = (5 ± 1)/2. So x = 2 or 3. Final Answer: 2, 3",
        "We can complete the square: (x - 5/2)^2 = 1/4, so x = 2 or 3. Final Answer: x = 2, 3",
        "Testing values: when x=2, x^2-5x+6=0. When x=3, x^2-5x+6=0. Final Answer: 2, 3",
        "By factoring: (x-2)(x-3)=0 gives x=2 or x=3. Final Answer: {2, 3}",
        "Solving x^2 - 5x + 6 = 0: factor to get (x-2)(x-3)=0. Final Answer: x = 2, x = 3",
        "We have x^2-5x+6=0. Factoring: (x-2)(x-3)=0. Therefore x = 2 or x = 3. Final Answer: 2,3",
        "Use quadratic formula with a=1, b=-5, c=6: x = (5±1)/2 = 2 or 3. Final Answer: x=2,3",
    ]
    
    diversity = compute_jaccard_diversity_for_group(responses)
    print(f"Responses: 8 different solution approaches to x^2 - 5x + 6 = 0")
    print(f"Diversity Score: {diversity:.4f}")
    print(f"Interpretation: {diversity * 100:.1f}% different from each other")
    print(f"\nNote: Despite reaching the same answer (x=2,3), the solution")
    print(f"      methods vary (factoring, quadratic formula, completing square, etc.)")


def example_4_with_reward():
    """Example: Show how diversity bonus affects final rewards"""
    print("\n" + "="*60)
    print("Example 4: Impact on Final Rewards")
    print("="*60)
    
    # Scenario A: Low diversity
    responses_low_diversity = [
        "The answer is 42",
        "The answer is 42",
        "The answer is 42",
        "The answer is 42",
        "The answer is 42",
        "The answer is 42",
        "The answer is 42",
        "The answer is 42",
    ]
    diversity_low = compute_jaccard_diversity_for_group(responses_low_diversity)
    
    # Scenario B: High diversity
    responses_high_diversity = [
        "Calculate: 6*7 = 42",
        "Sum of first 9 positive integers is 45, minus 3 equals 42",
        "The answer to life universe and everything is 42",
        "Forty-two or 42 in decimal notation",
        "Binary 101010 converts to decimal 42",
        "Six times seven yields fourty-two",
        "Result equals 42 after computation",
        "Final value obtained is 42",
    ]
    diversity_high = compute_jaccard_diversity_for_group(responses_high_diversity)
    
    # Assume all correct (base_reward = 1.0), diversity_weight = 0.1
    base_reward = 1.0
    diversity_weight = 0.1
    
    final_reward_low = base_reward + diversity_weight * diversity_low
    final_reward_high = base_reward + diversity_weight * diversity_high
    
    print(f"\nScenario A: Low Diversity Responses")
    print(f"  Diversity: {diversity_low:.4f}")
    print(f"  Base Reward: {base_reward:.2f} (correct answer)")
    print(f"  Diversity Bonus: {diversity_weight * diversity_low:.4f}")
    print(f"  Final Reward: {final_reward_low:.4f}")
    
    print(f"\nScenario B: High Diversity Responses")
    print(f"  Diversity: {diversity_high:.4f}")
    print(f"  Base Reward: {base_reward:.2f} (correct answer)")
    print(f"  Diversity Bonus: {diversity_weight * diversity_high:.4f}")
    print(f"  Final Reward: {final_reward_high:.4f}")
    
    print(f"\nReward Difference: {final_reward_high - final_reward_low:.4f}")
    print(f"Percentage Increase: {(final_reward_high / final_reward_low - 1) * 100:.2f}%")


def example_5_intersection_union():
    """Example: Show intersection/union calculation"""
    print("\n" + "="*60)
    print("Example 5: Intersection/Union Breakdown")
    print("="*60)
    
    responses = [
        "apple banana cherry",
        "apple banana date",
        "apple elderberry fig",
        "grape honeydew",
    ]
    
    print("Responses:")
    token_sets = []
    for i, resp in enumerate(responses, 1):
        tokens = tokenize_response(resp)
        token_sets.append(tokens)
        print(f"  {i}. {resp}")
        print(f"     Tokens: {tokens}")
    
    # Compute intersection (common to ALL)
    intersection = token_sets[0].copy()
    for token_set in token_sets[1:]:
        intersection = intersection.intersection(token_set)
    
    # Compute union (all unique tokens)
    union = set()
    for token_set in token_sets:
        union = union.union(token_set)
    
    jaccard_index = len(intersection) / len(union) if len(union) > 0 else 0.0
    diversity = 1.0 - jaccard_index
    
    print(f"\nIntersection (common to ALL): {intersection}")
    print(f"  Size: {len(intersection)}")
    print(f"\nUnion (all unique tokens): {union}")
    print(f"  Size: {len(union)}")
    print(f"\nJaccard Index: {len(intersection)} / {len(union)} = {jaccard_index:.4f}")
    print(f"Diversity (1 - Jaccard): {diversity:.4f}")
    
    # Verify with function
    computed_diversity = compute_jaccard_diversity_for_group(responses)
    print(f"Function result: {computed_diversity:.4f} (should match above)")


def main():
    """Run all examples"""
    print("\n" + "#"*60)
    print("#  JACCARD DIVERSITY CALCULATION EXAMPLES")
    print("#  Method: 1 - (intersection of all / union of all)")
    print("#"*60)
    
    example_1_identical_responses()
    example_2_completely_different()
    example_3_realistic_math()
    example_4_with_reward()
    example_5_intersection_union()
    
    print("\n" + "#"*60)
    print("#  KEY TAKEAWAYS")
    print("#"*60)
    print("""
1. Diversity = 1 - (tokens common to ALL / all unique tokens)
2. Diversity ranges from 0 (identical) to 1 (completely different)
3. With diversity_weight=0.1, the bonus ranges from 0 to 0.1
4. Accuracy remains the dominant signal (weight=1.0)
5. Diversity encourages exploring different solution approaches
6. The model still needs to be correct to get positive reward
""")


if __name__ == "__main__":
    main()

