import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Load the CSV files
df_0shot = pd.read_csv("/work/pi_wenlongzhao_umass_edu/25/kdasoju/Phi3.5_Router/phi_routing_results_log_probs_with_scores/diverse_few_shot/set1/0shot.csv")
df_11shot_set1 = pd.read_csv("/work/pi_wenlongzhao_umass_edu/25/kdasoju/Phi3.5_Router/phi_routing_results_log_probs_with_scores/diverse_few_shot/set1/11shot.csv")
df_11shot_set2 = pd.read_csv("/work/pi_wenlongzhao_umass_edu/25/kdasoju/Phi3.5_Router/phi_routing_results_log_probs_with_scores/diverse_few_shot/set2/11shot.csv")

# Model names
model_names = [
    "WizardLM/WizardLM-13B-V1.2",
    "claude-instant-v1",
    "claude-v1",
    "claude-v2",
    "gpt-3.5-turbo-1106",
    "gpt-4-1106-preview",
    "meta/code-llama-instruct-34b-chat",
    "meta/llama-2-70b-chat",
    "mistralai/mistral-7b-chat",
    "mistralai/mixtral-8x7b-chat",
    "zero-one-ai/Yi-34B-Chat"
]

# For phi_router tracking across shots
phi_router_points = []

# First: plot all models from 0-shot
avg_correctness = []
total_cost = []
labels = []

for model in model_names:
    avg_correctness.append(df_0shot[f"{model}_correctness"].mean())
    total_cost.append(df_0shot[f"{model}_cost"].sum())
    labels.append(model)

# Plotting
plt.figure(figsize=(12, 7))

# Add model points from 0-shot
for i in range(len(labels)):
    plt.scatter(total_cost[i], avg_correctness[i], label=labels[i], s=80)  # slightly bigger dots

# Add and store phi_router points for all three shots
for df, label in zip([df_0shot, df_11shot_set1, df_11shot_set2], ["0-shot", "11-shot_set1", "11-shot_set2"]):
    phi_avg = df["phi_correctness"].mean()
    phi_cost = df["phi_cost"].sum()
    phi_router_points.append((phi_cost, phi_avg))
    plt.scatter(phi_cost, phi_avg, color='red', s=90)
    plt.annotate(
        f"{label}",
        (phi_cost, phi_avg),
        xytext=(8, -10),
        textcoords="offset points",
        fontsize=20,  # Larger annotation text
        color='red'
    )

# Draw line connecting phi_router points
phi_costs, phi_scores = zip(*phi_router_points)
plt.plot(phi_costs, phi_scores, color='red', linestyle='--', linewidth=2, label="phi_router progression")

# Final plot settings
plt.title("Cost vs Performance: Phi Router (Few-shot) vs All Models", fontsize=24)
plt.xlabel("Total Cost", fontsize=24)
plt.ylabel("Average Performance", fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=20, title="Models", title_fontsize=22)  # Bigger legend


# Save with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"cost_vs_performance_allshots_logprobs_{timestamp}.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"Plot saved as {output_file}")
