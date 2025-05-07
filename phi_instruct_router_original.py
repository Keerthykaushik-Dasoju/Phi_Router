from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import gc
from datetime import datetime

# Clean up GPU memory
torch.cuda.empty_cache()
gc.collect()

# Load datasets
train_data = pd.read_csv("/work/pi_wenlongzhao_umass_edu/25/data/routerbench/routerbench_0shot_train.csv")
test_data = pd.read_csv("/work/pi_wenlongzhao_umass_edu/25/data/routerbench/routerbench_0shot_test.csv")

# Sample 11 few-shot examples
few_shot_data = train_data.sample(n=11, random_state=42)

# Build few-shot message history
messages = [{
    "role": "system",
    "content": (
        "You are an intelligent LLM router that selects the optimal model to answer a user's prompt.\n"
        "You are shown multiple models with their correctness (0 = wrong, 1 = correct) and cost (0.0 - 1.0).\n"
        "Always choose a model with correctness 1. If multiple are correct, pick the one with the lowest cost.\n"
        "Respond with only the model name.\n\n"
        "Example:\n"
        "Prompt: What is 2 + 2?\n"
        "Model A — correctness: 1, cost: 0.9\n"
        "Model B — correctness: 0, cost: 0.1\n"
        "Model C — correctness: 1, cost: 0.3\n"
        "Correct choice: Model C"
    )
}]

# Add few-shot examples (with correctness + cost)
for _, row in few_shot_data.iterrows():
    prompt = row['prompt']
    print(prompt)
    print(row['oracle_model_to_route_to'])
    print()
    model_stats = [
        f"WizardLM/WizardLM-13B-V1.2 — correctness: {row['WizardLM/WizardLM-13B-V1.2']}, cost: {row['WizardLM/WizardLM-13B-V1.2|total_cost']}",
        f"claude-instant-v1 — correctness: {row['claude-instant-v1']}, cost: {row['claude-instant-v1|total_cost']}",
        f"claude-v1 — correctness: {row['claude-v1']}, cost: {row['claude-v1|total_cost']}",
        f"claude-v2 — correctness: {row['claude-v2']}, cost: {row['claude-v2|total_cost']}",
        f"gpt-3.5-turbo-1106 — correctness: {row['gpt-3.5-turbo-1106']}, cost: {row['gpt-3.5-turbo-1106|total_cost']}",
        f"gpt-4-1106-preview — correctness: {row['gpt-4-1106-preview']}, cost: {row['gpt-4-1106-preview|total_cost']}",
        f"meta/code-llama-instruct-34b-chat — correctness: {row['meta/code-llama-instruct-34b-chat']}, cost: {row['meta/code-llama-instruct-34b-chat|total_cost']}",
        f"meta/llama-2-70b-chat — correctness: {row['meta/llama-2-70b-chat']}, cost: {row['meta/llama-2-70b-chat|total_cost']}",
        f"mistralai/mistral-7b-chat — correctness: {row['mistralai/mistral-7b-chat']}, cost: {row['mistralai/mistral-7b-chat|total_cost']}",
        f"mistralai/mixtral-8x7b-chat — correctness: {row['mistralai/mixtral-8x7b-chat']}, cost: {row['mistralai/mixtral-8x7b-chat|total_cost']}",
        f"zero-one-ai/Yi-34B-Chat — correctness: {row['zero-one-ai/Yi-34B-Chat']}, cost: {row['zero-one-ai/Yi-34B-Chat|total_cost']}"
    ]
    messages.append({"role": "user", "content": f"Prompt: {prompt}\n" + "\n".join(model_stats)})
    messages.append({"role": "assistant", "content": row['oracle_model_to_route_to']})

# Load model normally on GPU (no quantization)
model_path = "/work/pi_wenlongzhao_umass_edu/25/kdasoju/phi3_5_mini_instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,       # Optional: float16 for memory savings
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Generation function
def chat_completion(input_messages, max_new_tokens=50):
    prompt = tokenizer.apply_chat_template(input_messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0
        )
    return tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

# Store results for saving
all_outputs = []

test_data = test_data.sample(n=0, random_state=42)
# Run on full test set (prompt only, no cost/correctness)
for _, row in test_data.iterrows():
    prompt = row['prompt']
    sample_id = row['sample_id']

    test_messages = messages + [{
        "role": "user",
        "content": (
            f"Prompt: {prompt}\n\n"
            f"Which model is best suited to answer this prompt?\n"
            f"Respond with only the model name.\n\n"
            f"Available models: WizardLM/WizardLM-13B-V1.2, claude-instant-v1, claude-v1, claude-v2, "
            f"gpt-3.5-turbo-1106, gpt-4-1106-preview, meta/code-llama-instruct-34b-chat, meta/llama-2-70b-chat, "
            f"mistralai/mistral-7b-chat, mistralai/mixtral-8x7b-chat, zero-one-ai/Yi-34B-Chat"
        )
    }]

    result = chat_completion(test_messages)

    # print(">> Phi output:", result)
    # print(">> Oracle Answer:", row['oracle_model_to_route_to'])
    # print("-" * 70)

    all_outputs.append({
        "sample_id": sample_id,
        "phi_prediction": result,
    })

    torch.cuda.empty_cache()
    gc.collect()

output_df = pd.DataFrame(all_outputs)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"phi_routing_results{timestamp}.csv"
output_df.to_csv(output_file, index=False)
print(f"✅ Saved results to {output_file}")