from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import pandas as pd
import random
import gc
import torch

torch.cuda.empty_cache()
gc.collect()

# Load the CSV file
data = pd.read_csv("/work/pi_wenlongzhao_umass_edu/25/mghantasala/routerbench_0shot_train.csv")

# Sample 20 random rows from the dataframe
sampled_data = data.sample(n=11, random_state=42)

# Prepare the message format for each sample
messages = []
tmp = ""
for index, row in sampled_data.iterrows(): 
    prompt = row['prompt']
    tmp =prompt
    eval_name = row['eval_name']
    
    model_performances = [
        f"Model WizardLM/WizardLM-13B-V1.2 : <parameters> correctness:{row['WizardLM/WizardLM-13B-V1.2']} cost:{row['WizardLM/WizardLM-13B-V1.2|total_cost']}</parameters>",
        f"Model claude-instant-v1 : <parameters> correctness:{row['claude-instant-v1']} cost:{row['claude-instant-v1|total_cost']}</parameters>",
        f"Model claude-v1 : <parameters> correctness:{row['claude-v1']} cost:{row['claude-v1|total_cost']}</parameters>",
        f"Model claude-v2 : <parameters> correctness:{row['claude-v2']} cost:{row['claude-v2|total_cost']}</parameters>",
        f"Model gpt-3.5-turbo-1106 : <parameters> correctness:{row['gpt-3.5-turbo-1106']} cost:{row['gpt-3.5-turbo-1106|total_cost']}</parameters>",
        f"Model gpt-4-1106-preview : <parameters> correctness:{row['gpt-4-1106-preview']} cost:{row['gpt-4-1106-preview|total_cost']}</parameters>",
        f"Model meta/code-llama-instruct-34b-chat : <parameters> correctness:{row['meta/code-llama-instruct-34b-chat']} cost:{row['meta/code-llama-instruct-34b-chat|total_cost']}</parameters>",
        f"Model meta/llama-2-70b-chat : <parameters> correctness:{row['meta/llama-2-70b-chat']} cost:{row['meta/llama-2-70b-chat|total_cost']}</parameters>",
        f"Model mistralai/mistral-7b-chat : <parameters> correctness:{row['mistralai/mistral-7b-chat']} cost:{row['mistralai/mistral-7b-chat|total_cost']}</parameters>",
        f"Model mistralai/mixtral-8x7b-chat : <parameters> correctness:{row['mistralai/mixtral-8x7b-chat']} cost:{row['mistralai/mixtral-8x7b-chat|total_cost']}</parameters>",
        f"Model zero-one-ai/Yi-34B-Chat : <parameters> correctness:{row['zero-one-ai/Yi-34B-Chat']} cost:{row['zero-one-ai/Yi-34B-Chat|total_cost']}</parameters>"
    ]
    
    # Example message
    messages.append(
        {
            "role": "system",
            "content": f"You are to act as a LLM router and select the best model to answer the user's question. <context> available models are: WizardLM/WizardLM-13B-V1.2, claude-instant-v1, claude-v1, claude-v2, gpt-3.5-turbo-1106, gpt-4-1106-preview, meta/code-llama-instruct-34b-chat, meta/llama-2-70b-chat, mistralai/mistral-7b-chat, mistralai/mixtral-8x7b-chat, zero-one-ai/Yi-34B-Chat  </context> <parameters> correctness:(0/1) 0 represents the model got it wrong and 1 represents the model got it right, cost:(0-1) cost for tokens </parameters>"
        }
    )
    messages.append(
        {
            "role": "user",
            "content": f"{prompt}  {''.join(model_performances)}"
        }
    )
    messages.append(
        {
            "role": "assistant",
            "content": f"Sure! The model to be routed is {row['oracle_model_to_route_to']}"
        }
    )

local_model_path = "/work/pi_wenlongzhao_umass_edu/25/kdasoju/phi3_5_mini_instruct"

tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(local_model_path, trust_remote_code=True)
# model.to("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    local_model_path, 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
)
# tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")


datatest = pd.read_csv("/work/pi_wenlongzhao_umass_edu/25/mghantasala/routerbench_0shot_test.csv")
test_sampled_data = datatest.sample(n=10, random_state=42)


pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 50,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

# REMOVE the pipeline block

# Add this function to handle generation:
def chat_completion(messages, max_new_tokens=50):
    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()



for index, row in test_sampled_data.iterrows():
    prompt = row['prompt']
    eval_name = row['eval_name']

    messages.append(
        {
            "role": "user",
            "content": f"For this given prompt: {prompt}\n which model do you think would be the best to route it to without any explanation? **expected answer: model name** **example: mistralai/mixtral-8x7b-chat**"
        }
    )
    output = pipe(messages, **generation_args)
    # result = chat_completion(messages)
    # print(">> Phi output: " + result)   
    # print(f">> Prompt: {row['prompt'][:80]}...")
    print(">> Phi output: " +output[0]['generated_text'])
    print(f">> Oracle Answer: {row['oracle_model_to_route_to']}\n{'-'*70}")
    torch.cuda.empty_cache()
    gc.collect()
    messages.pop()