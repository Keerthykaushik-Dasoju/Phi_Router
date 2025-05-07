from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import gc
from datetime import datetime
import torch.nn.functional as F
import argparse
from torch.cuda import OutOfMemoryError
import os

# Clean up GPU memory
torch.cuda.empty_cache()
gc.collect()

CHECKPOINT_EVERY = 5
output_file = "phi_routing_results_log_probs_11shot.csv"

parser = argparse.ArgumentParser(description="LLM Router Configuration")

# Default: True, but user can override with --no-pick_diverse_prompts
parser.add_argument(
    "--pick_diverse_prompts",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable or disable diverse prompt selection (default: True)"
)

parser.add_argument(
    "--required_num_of_shots",
    type=int,
    required=True,
    help="Number of few-shot examples to use (mandatory)"
)

args = parser.parse_args()

# Load datasets
train_data = pd.read_csv("/work/pi_wenlongzhao_umass_edu/25/data/routerbench/routerbench_0shot_train.csv")
test_data = pd.read_csv("/work/pi_wenlongzhao_umass_edu/25/data/routerbench/routerbench_0shot_test.csv")

# Sample few-shot examples
few_shot_data = train_data.sample(n=1000, random_state=42)

system_message_zero_shot = {
    "role": "system",
    "content": """You are an intelligent LLM router that selects the most suitable model to answer a user's prompt.
        Each model has different capabilities and cost. More capable models generally perform better on harder prompts but may cost more.
        Given a user prompt, choose the model that provides the best balance of accuracy and cost.
        You are not given any explicit correctness or cost values — you must decide based on your understanding of the prompt and the model names.
        Respond with only the model name."""
}

system_message_few_shot = {
    "role": "system",
    "content": """You are an intelligent LLM router that selects the best model to answer a user's prompt.
        Each model has a correctness score (1 = correct, 0 = incorrect) and an associated cost (between 0.0 and 1.0).
        Your goal is to always choose a model with correctness 1. If multiple models are correct, select the one with the lowest cost.
        You will now be shown several examples where both correctness and cost are provided, along with the correct model to route to.

        At test time, you will only see the prompt and a list of available models — no correctness or cost scores will be given.
        From the examples you have seen, you are expected to learn which models perform well for different types of queries and how to balance performance with cost.
        Use this understanding to choose the most appropriate model.

        Respond with only the model name.

        Example:
        Prompt: What is 2 + 2?
        Model A — correctness: 1, cost: 0.9  
        Model B — correctness: 0, cost: 0.1  
        Model C — correctness: 1, cost: 0.3  
        Correct choice: Model C
        """
}

# Build few-shot message history
messages = [system_message_few_shot]

candidate_models_set = {"WizardLM/WizardLM-13B-V1.2", "claude-instant-v1", "claude-v1", "claude-v2",
    "gpt-3.5-turbo-1106", "gpt-4-1106-preview", "meta/code-llama-instruct-34b-chat",
    "meta/llama-2-70b-chat", "mistralai/mistral-7b-chat",
    "mistralai/mixtral-8x7b-chat", "zero-one-ai/Yi-34B-Chat"}


curr_num_of_shots = 0
# Use the parsed arguments
pick_diverse_prompts = args.pick_diverse_prompts
required_num_of_shots = args.required_num_of_shots

print(f"Diverse prompt selection: {pick_diverse_prompts}")
print(f"Number of shots: {required_num_of_shots}")


# Add few-shot examples
for _, row in few_shot_data.iterrows():
    prompt = row['prompt']
    oracle_model = row['oracle_model_to_route_to']
    if curr_num_of_shots == required_num_of_shots:
        break
    if oracle_model in candidate_models_set:
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
        curr_num_of_shots += 1
        if pick_diverse_prompts:
            candidate_models_set.remove(oracle_model)

# Load model
model_path = "/work/pi_wenlongzhao_umass_edu/25/kdasoju/phi3_5_mini_instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Efficient log-prob scorer (batched)
def score_model_names_batched(prompt_text, candidates):
    full_texts = [prompt_text + candidate for candidate in candidates]
    tokenized = tokenizer(full_texts, return_tensors="pt", padding=True)
    input_ids = tokenized.input_ids.to(model.device)
    attention_mask = tokenized.attention_mask.to(model.device)

    model.eval()

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)

    scores = {}
    for i, candidate in enumerate(candidates):
        full_input_ids = input_ids[i]
        candidate_ids = tokenizer(candidate, return_tensors="pt", add_special_tokens=False).input_ids[0].to(model.device)
        offset = len(full_input_ids) - len(candidate_ids)

        candidate_logprob = 0.0
        for j, token_id in enumerate(candidate_ids):
            candidate_logprob += log_probs[i, offset + j - 1, token_id].item()

        scores[candidate] = candidate_logprob

    return scores

def score_model_names(prompt_text, candidates):
    scores = {}

    for candidate in candidates:
        full_input = prompt_text + candidate
        input_ids = tokenizer(full_input, return_tensors="pt").input_ids.to(model.device)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        log_probs = F.log_softmax(logits, dim=-1)

        # Tokenize the candidate string to get its ids
        candidate_ids = tokenizer(candidate, return_tensors="pt", add_special_tokens=False).input_ids[0].to(model.device)
        offset = input_ids.shape[1] - candidate_ids.shape[0]

        candidate_logprob = 0.0
        for j, token_id in enumerate(candidate_ids):
            candidate_logprob += log_probs[0, offset + j - 1, token_id].item()

        scores[candidate] = candidate_logprob

        torch.cuda.empty_cache()
        gc.collect()

    return scores


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

processed_ids = set()
if os.path.exists(output_file):
    existing_df = pd.read_csv(output_file)
    processed_ids = set(existing_df["sample_id"])
    print(f"🔁 Resuming from checkpoint. Already processed {len(processed_ids)} samples.")
else:
    existing_df = pd.DataFrame()

# Run evaluation
all_outputs = []

test_data = test_data.sample(n=1000, random_state=42)
candidate_models = [
    "WizardLM/WizardLM-13B-V1.2", "claude-instant-v1", "claude-v1", "claude-v2",
    "gpt-3.5-turbo-1106", "gpt-4-1106-preview", "meta/code-llama-instruct-34b-chat",
    "meta/llama-2-70b-chat", "mistralai/mistral-7b-chat",
    "mistralai/mixtral-8x7b-chat", "zero-one-ai/Yi-34B-Chat"
]
count = 0
for _, row in test_data.iterrows():
    count += 1
    print(count)
    prompt = row['prompt']
    sample_id = row['sample_id']
    if sample_id in processed_ids:
        continue  # Skip already processed

    test_messages = messages + [{
        "role": "user",
        "content": (
            f"Prompt: {prompt}\n\n"
            f"Which model is best suited to answer this prompt?\n"
            f"Respond with only the model name.\n\n"
            f"Available models: {', '.join(candidate_models)}"
        )
    }]

    try:
        result = chat_completion(test_messages)

        logprob_prompt = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
        logprob_scores = score_model_names(logprob_prompt, candidate_models)
        logprob_best_model = max(logprob_scores, key=logprob_scores.get)

        all_outputs.append({
            "sample_id": sample_id,
            "phi_prediction": logprob_best_model,
            "phi_response": result,
            "oracle_model_to_route": row['oracle_model_to_route_to'],
        })

        # Save every CHECKPOINT_EVERY samples
        if len(all_outputs) >= CHECKPOINT_EVERY:
            df = pd.DataFrame(all_outputs)
            df.to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file))
            print(f"💾 Saved checkpoint with {len(all_outputs)} entries.")
            processed_ids.update(df["sample_id"])
            all_outputs = []  # Reset for next batch

    except OutOfMemoryError:
        print(f"Skipped sample_id {sample_id} due to CUDA OOM.")
    finally:
        torch.cuda.empty_cache()
        gc.collect()

if all_outputs:
    df = pd.DataFrame(all_outputs)
    df.to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file))
    print(f"💾 Final checkpoint saved with {len(all_outputs)} entries.")


# output_df = pd.DataFrame(all_outputs)
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# output_df.to_csv(output_file, index=False)
# print(f"✅ Saved results to {output_file}")
