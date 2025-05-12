import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from matplotlib.font_manager import FontProperties

bold_title_font = FontProperties(weight='bold', size=22)

parser = argparse.ArgumentParser(description="Match Predictions Configuration")

parser.add_argument(
    "--domain",
    type=str,
    required=True,
    help="Number of few-shot examples to use (mandatory)"
)

args = parser.parse_args()

# Use the parsed arguments
domain = args.domain
# === STEP 1: Define all file paths and labels here ===
file_paths = [
    (f"/work/pi_wenlongzhao_umass_edu/25/kdasoju/Phi3.5_Router/phi_routing_results_with_scores/5_models_used/log_probs/domain_wise/{domain}/0shot.csv", "0-shot"),
    (f"/work/pi_wenlongzhao_umass_edu/25/kdasoju/Phi3.5_Router/phi_routing_results_with_scores/5_models_used/log_probs/domain_wise/{domain}/5shot_random.csv", "r5-shot"),
    (f"/work/pi_wenlongzhao_umass_edu/25/kdasoju/Phi3.5_Router/phi_routing_results_with_scores/5_models_used/log_probs/domain_wise/{domain}/5shot_diverse.csv", "d5-shot"),
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

plt.figure(figsize=(12, 9))

for i in range(len(labels)):
    plt.scatter(total_cost[i], avg_correctness[i], label=labels[i], s=120)

# === STEP 5: Phi router progression ===
phi_router_points = []
for df, label in dataframes:
    phi_avg = df["phi_correctness"].mean()
    phi_cost = df["phi_cost"].sum()
    phi_router_points.append((phi_cost, phi_avg))
    plt.scatter(phi_cost, phi_avg, color='red', s=90)
    plt.annotate(label, (phi_cost, phi_avg), xytext=(8, -10), textcoords="offset points", fontsize=14, color='red')

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
plt.scatter(oracle_total_cost, oracle_avg_correctness, color='deepskyblue', marker='*', s=200, label="Oracle (best per sample)")
plt.annotate("", (oracle_total_cost, oracle_avg_correctness), xytext=(8, -10), textcoords="offset points", fontsize=12, color='gold')

# === Final plot formatting ===
phi_costs, phi_scores = zip(*phi_router_points)
plt.plot(phi_costs, phi_scores, color='red', linestyle='--', linewidth=2, label="phi_router progression")

plt.title(f"Cost vs Performance: Phi Router ({domain})", fontsize=20, fontweight="bold")
plt.xlabel("Total Cost", fontsize=20, fontweight="bold")
plt.ylabel("Average Performance", fontsize=20, fontweight="bold")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)
# plt.tight_layout()
plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=3,  # controls number of items per row; adjust based on how many you have
    fontsize=20,
    title="Models",
    title_fontproperties=bold_title_font
)
plt.subplots_adjust(left=0.15, right=0.85, top=0.75, bottom=0.25)

# Save
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"cost_vs_performance_{domain}.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"Plot saved as {output_file}")
