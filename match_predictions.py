import pandas as pd
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(description="Match Predictions Configuration")

parser.add_argument(
    "--file_suffix",
    type=str,
    required=True,
    help="Number of few-shot examples to use (mandatory)"
)

args = parser.parse_args()

# Use the parsed arguments
file_suffix = args.file_suffix

# Load both files
test_df = pd.read_csv("/work/pi_wenlongzhao_umass_edu/25/data/routerbench/routerbench_0shot_test.csv")
router_df = pd.read_csv(f"/work/pi_wenlongzhao_umass_edu/25/kdasoju/Phi3.5_Router/phi_routing_results_log_probs_{file_suffix}.csv")

# All model names (11 total)
model_names = [
    "mistralai/mistral-7b-chat",
    "WizardLM/WizardLM-13B-V1.2",
    "mistralai/mixtral-8x7b-chat",
    'meta/code-llama-instruct-34b-chat',
    "gpt-4-1106-preview",
]

# Initialize list to collect final rows
final_rows = []
not_a_valid_model_response_count = 0
test_row_not_found_count = 0


for _, row in router_df.iterrows():
    sample_id = row["sample_id"]
    predicted_model = row["phi_prediction"]

    # Match test sample row
    test_row = test_df[test_df["sample_id"] == sample_id]

    if test_row.empty:
        test_row_not_found_count += 1
        continue  # Skip if sample_id not found in test set

    test_row = test_row.iloc[0]

    # Default values
    phi_correctness = 0.0
    phi_cost = 0.0

    if predicted_model not in test_row.index:
        not_a_valid_model_response_count += 1
    else:
        # Extract phi prediction stats
        phi_correctness = test_row[predicted_model]
        phi_cost = test_row.get(f"{predicted_model}|total_cost", 0.0)

    # Extract all 11 model performances and costs
    row_data = {
        "sample_id": sample_id,
        "phi_prediction": predicted_model,
        "phi_correctness": phi_correctness,
        "phi_cost": phi_cost
    }

    for model in model_names:
        row_data[f"{model}_correctness"] = test_row.get(model, None)
        row_data[f"{model}_cost"] = test_row.get(f"{model}|total_cost", None)

    final_rows.append(row_data)

# Convert to DataFrame
final_df = pd.DataFrame(final_rows)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# output_file = f"phi_routing_results_log_probs_with_scores_11shot_{timestamp}.csv"
output_file = f"phi_routing_results_log_probs_with_scores_{file_suffix}.csv"
# Save the result
final_df.to_csv(output_file, index=False)

# Show stats
print(f"Saved final results to {output_file }")
print(f"Test row not found count: {test_row_not_found_count}")
print(f"Not a valid model response count: {not_a_valid_model_response_count}")
