import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Load the merged CSV
df = pd.read_csv("/work/pi_wenlongzhao_umass_edu/25/kdasoju/Phi3.5_Router/phi_routing_results_log_probs_with_scores_0shot.csv")

# List of model names (11 total)
model_names = [
    "mistralai/mistral-7b-chat",
    "WizardLM/WizardLM-13B-V1.2",
    "mistralai/mixtral-8x7b-chat",
    'meta/code-llama-instruct-34b-chat',
    "gpt-4-1106-preview",
]

avg_correctness = []
total_cost = []
labels = []

# Add router stats
avg_correctness.append(df["phi_correctness"].mean())
total_cost.append(df["phi_cost"].sum())
labels.append("phi_router")

# Add each model stats
for model in model_names:
    avg_correctness.append(df[f"{model}_correctness"].mean())
    total_cost.append(df[f"{model}_cost"].sum())
    labels.append(model)

print(avg_correctness)
print(total_cost)
# Plot
plt.figure(figsize=(12, 7))
for i in range(len(labels)):
    if labels[i] == "phi_router":
        plt.scatter(total_cost[i], avg_correctness[i], color='red', s=70)
        plt.annotate(
            "phi_router",
            (total_cost[i], avg_correctness[i]),
            xytext=(8, -10),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            color="red"
        )
    else:
        plt.scatter(total_cost[i], avg_correctness[i], label=labels[i], s=60)

plt.title("Cost vs Performance: Phi Router vs All Models")
plt.xlabel("Total Cost")
plt.ylabel("Average Performance")
plt.grid(True)
plt.tight_layout()

# Show legend outside the plot (can also change loc="lower center" for below)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9, title="Models")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"cost_vs_performance_11shot_{timestamp}.png"
# Save the plot
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"Plot saved as {output_file}")
