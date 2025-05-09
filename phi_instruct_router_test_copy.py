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
output_file = "phi_routing_results_log_probs_5shot_test2_diverse.csv"

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

candidate_models = [
    "mistralai/mistral-7b-chat",
    "WizardLM/WizardLM-13B-V1.2",
    "mistralai/mixtral-8x7b-chat",
    'meta/code-llama-instruct-34b-chat',
    "gpt-4-1106-preview",
]

candidate_models_set = set(candidate_models)


curr_num_of_shots = 0
# Use the parsed arguments
pick_diverse_prompts = args.pick_diverse_prompts
required_num_of_shots = args.required_num_of_shots

print(f"Diverse prompt selection: {pick_diverse_prompts}")
print(f"Number of shots: {required_num_of_shots}")


# Add few-shot examples
for _, row in few_shot_data.iterrows():
    prompt = row['prompt']
    sample_id = row['sample_id']
    oracle_model = row['oracle_model_to_route_to']
    if curr_num_of_shots == required_num_of_shots:
        break
    if oracle_model in candidate_models_set:
        model_stats = [
            f"{model} — correctness: {row[model]}, cost: {row[f'{model}|total_cost']}"
            for model in candidate_models
        ]
        messages.append({
            "role": "user",
            "content": (
                f"Prompt: {prompt}\n\n"
                f"Following are the correctness and cost values for each available model for the above prompt:\n"
                + "\n".join(model_stats)
                + "\n\nYou are an intelligent LLM router. You must understand the prompt and evaluate each model based on both correctness and cost.\n"
                + "Always choose a model with correctness = 1 (i.e., one that answers the prompt correctly).\n"
                + "If multiple models are correct, select the one with the lowest cost.\n"
                + "What is the correct model to route this prompt to?\n"
                + "Respond with only the model name."
            )
        })
        messages.append({"role": "assistant", "content": row['oracle_model_to_route_to']})
        curr_num_of_shots += 1
        print(oracle_model)
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

# test_data = test_data.sample(n=10, random_state=42)
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
            f"You are an intelligent LLM router.\n"
            f"Based on your understanding of the prompt and the capabilities of the models listed below,\n"
            f"Choose the most suitable model to answer this prompt.\n\n"
            f"Respond with only the model name, exactly as listed below — no explanation or extra text.\n"
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
