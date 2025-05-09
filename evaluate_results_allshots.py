import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# === STEP 1: Define all file paths and labels here ===
file_paths = [
    ("/work/pi_wenlongzhao_umass_edu/25/kdasoju/Phi3.5_Router/phi_routing_results_log_probs_with_scores_5shot_pos0.csv", "d5-shot_pos0"),
    ("/work/pi_wenlongzhao_umass_edu/25/kdasoju/Phi3.5_Router/phi_routing_results_log_probs_with_scores_5shot_pos1.csv", "d5-shot_pos1"),
    ("/work/pi_wenlongzhao_umass_edu/25/kdasoju/Phi3.5_Router/phi_routing_results_log_probs_with_scores_5shot_pos2.csv", "d5-shot_pos2"),
    ("/work/pi_wenlongzhao_umass_edu/25/kdasoju/Phi3.5_Router/phi_routing_results_log_probs_with_scores_5shot_pos3.csv", "d5-shot_pos3"),
    ("/work/pi_wenlongzhao_umass_edu/25/kdasoju/Phi3.5_Router/phi_routing_results_log_probs_with_scores_5shot_pos4.csv", "d5-shot_pos4"),
]

# === STEP 2: Load all dataframes dynamically ===
dataframes = [(pd.read_csv(path), label) for path, label in file_paths]

# Use the first file for computing model baselines and oracle
base_df, _ = dataframes[0]

# === STEP 3: Model names ===
model_names = [
    "mistralai/mistral-7b-chat",
    "WizardLM/WizardLM-13B-V1.2",
    "mistralai/mixtral-8x7b-chat",
    "meta/code-llama-instruct-34b-chat",
    "gpt-4-1106-preview",
]

# === STEP 4: Plotting all models' baseline ===
avg_correctness = [base_df[f"{model}_correctness"].mean() for model in model_names]
total_cost = [base_df[f"{model}_cost"].sum() for model in model_names]
labels = model_names

plt.figure(figsize=(12, 7))

for i in range(len(labels)):
    plt.scatter(total_cost[i], avg_correctness[i], label=labels[i], s=80)

# === STEP 5: Phi router progression ===
phi_router_points = []
for df, label in dataframes:
    phi_avg = df["phi_correctness"].mean()
    phi_cost = df["phi_cost"].sum()
    phi_router_points.append((phi_cost, phi_avg))
    plt.scatter(phi_cost, phi_avg, color='red', s=90)
    plt.annotate(label, (phi_cost, phi_avg), xytext=(8, -10), textcoords="offset points", fontsize=12, color='red')

# === STEP 6: Oracle model calculation ===
def get_oracle_model(row):
    max_correctness = max([row[f"{model}_correctness"] for model in model_names])
    candidates = [
        (model, row[f"{model}_cost"])
        for model in model_names
        if row[f"{model}_correctness"] == max_correctness
    ]
    return min(candidates, key=lambda x: x[1])[0]

# Add oracle columns to the base_df only
base_df["oracle_model"] = base_df.apply(get_oracle_model, axis=1)
base_df["oracle_correctness"] = base_df.apply(lambda row: row[f"{row['oracle_model']}_correctness"], axis=1)
base_df["oracle_cost"] = base_df.apply(lambda row: row[f"{row['oracle_model']}_cost"], axis=1)

oracle_avg_correctness = base_df["oracle_correctness"].mean()
oracle_total_cost = base_df["oracle_cost"].sum()

# === STEP 7: Plot Oracle ===
plt.scatter(oracle_total_cost, oracle_avg_correctness, color='gold', marker='*', s=200, label="Oracle (best per sample)")
plt.annotate("Oracle", (oracle_total_cost, oracle_avg_correctness), xytext=(8, -10), textcoords="offset points", fontsize=12, color='gold')

# === Final plot formatting ===
phi_costs, phi_scores = zip(*phi_router_points)
plt.plot(phi_costs, phi_scores, color='red', linestyle='--', linewidth=2, label="phi_router progression")

plt.title("Cost vs Performance: Phi Router (Few-shot) vs All Models", fontsize=24)
plt.xlabel("Total Cost", fontsize=24)
plt.ylabel("Average Performance", fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=20, title="Models", title_fontsize=22)

# Save
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"cost_vs_performance_allshots_logprobs_{timestamp}.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"Plot saved as {output_file}")
