import pandas as pd

# Load the file
file_path = "/work/pi_wenlongzhao_umass_edu/25/kdasoju/Phi3.5_Router/random_5shot/phi_routing_results_log_probs_with_scores_5shot.csv"  # Replace with your actual file path
df = pd.read_csv(file_path)

# List of models
models = [
    "mistralai/mistral-7b-chat",
    "WizardLM/WizardLM-13B-V1.2",
    "mistralai/mixtral-8x7b-chat",
    "meta/code-llama-instruct-34b-chat",
    "gpt-4-1106-preview",
]

# Prepare correctness and cost columns
correctness_cols = [f"{model}_correctness" for model in models]
cost_cols = [f"{model}_cost" for model in models]

# Function to compute oracle model for a row
def get_oracle_model(row):
    max_correctness = max(row[col] for col in correctness_cols)
    candidates = [
        (model, row[f"{model}_cost"])
        for model in models
        if row[f"{model}_correctness"] == max_correctness
    ]
    return min(candidates, key=lambda x: x[1])[0]

# Apply function
df["oracle_model"] = df.apply(get_oracle_model, axis=1)

# Save back to same file (overwrites the original)
df.to_csv(file_path, index=False)
print(f"oracle_model column added and saved to {file_path}")
