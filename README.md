# Activate conda env using below command before running this router
conda activate /work/pi_wenlongzhao_umass_edu/25/kdasoju/Phi3.5_Router/phi3.5_env 

1) prompt = """Instruction: Classify the sentiment of the following review as Positive, Negative, or Neutral.

Input: The movie had stunning visuals but a weak storyline and poor acting.

Response:
"""

2) prompt = """
Model 1:
Query: What is the capital of France?
Response: Paris
Reward: 1.0

Query: What is 2 + 2?
Response: 4
Reward: 1.0

---

Model 2:
Query: What is the capital of France?
Response: Paris is a famous city.
Reward: 0.8

Query: What is 2 + 2?
Response: I think it's 5.
Reward: 0.3

---

New Query: What is 5 * 6?
Target Reward: 0.2

Choose the best model (just give me which model is most suitable for the above query based on its reward factor distribution either model 1 or model 2) with no explanation:
"""

3) prompt = """
Model 1:
Query: What is the capital of France?
Response: Paris
Reward: 1.0

Query: What is 2 + 2?
Response: 4
Reward: 1.0

---

Model 2:
Query: What is the capital of France?
Response: Paris is a famous city.
Reward: 0.8

Query: What is 2 + 2?
Response: I think it's 5.
Reward: 0.3

---

New Query: What is 3 + 3?
Target Reward: 0.2

Select the model whose past responses align best with the target reward behavior. Just answer with "Model 1" or "Model 2" — no explanation:
"""


Latest Tried Prompts:

1. # 🧠 Chat-formatted prompt preserving original logic
prompt = """
<|system|>
You are an evaluator tasked with selecting which model's response aligns best with a target behavior based on reward scores.

Each response is followed by a numerical reward value between 0.0 and 1.0:
- Higher rewards (closer to 1.0) indicate responses that are more correct, relevant, and aligned with human preferences.
- Lower rewards (closer to 0.0) suggest responses that are incorrect, confusing, or misaligned with expectations.

Based on previous behavior of models shown below, your task is to decide which model would likely respond in a way that aligns with the given target reward for a new query, specifically after finding the similar context for the target query select the model whose reward is close to the target reward (for example 0.4 is closer to 0.1 not 1.0).

Just output the name of the best matching model — either "gpt-3.5-turbo-1106" or "meta/llama-2-70b-chat" — with no explanation.

<|user|>
# 📚 Past examples
gpt-3.5-turbo-1106:
Query: 'x+y = 4z, x*y = 4z^2, express x-y in z', 'Express z-x in y'
Response: 'To express x-y in terms of z, we can solve for x and y in terms of z from the given equations...
Reward: 0.5

meta/llama-2-70b-chat:
Query: 'x+y = 4z, x*y = 4z^2, express x-y in z', 'Express z-x in y'
Response: 'We are given two equations...
Reward: 0.1

gpt-3.5-turbo-1106:
Query: When a number is divided by 10, the remainder is 4. What is the remainder when twice the number is divided by 4?', 'What about when twice the number is divided by 5?'
Response: 'When a number is divided by 10, the remainder is 4...
Reward: 1.0

meta/llama-2-70b-chat:
Query: When a number is divided by 10, the remainder is 4. What is the remainder when twice the number is divided by 4?', 'What about when twice the number is divided by 5?'
Response: 'When a number is divided by 10, the remainder is 4...
Reward: 0.1

# ❓ New Query for Evaluation
Which model is more likely to give a response matching the following context?

Target Query: "When a number is divided by 10, the remainder is 3. What is the remainder when twice the number is divided by 3?', 'What about when twice the number is divided by 7?'"
Target Reward: 0.2

<|assistant|>
"""

2. # 🧠 Chat-formatted prompt preserving original logic
prompt = """
<|system|>
You are an evaluator tasked with selecting which model's response aligns best with a target behavior based on reward scores.

Each response is followed by a numerical reward value between 0.0 and 1.0:
- Higher rewards (closer to 1.0) indicate responses that are more correct, relevant, and aligned with human preferences.
- Lower rewards (closer to 0.0) suggest responses that are incorrect, confusing, or misaligned with expectations.

Based on previous behavior of models shown below, your task is to decide which model would likely respond in a way that aligns with the given target reward for a new query.

Just output the name of the best matching model — either "gpt-3.5-turbo-1106" or "meta/llama-2-70b-chat" — with no explanation.

<|user|>
# 📚 Past examples
gpt-3.5-turbo-1106:
Query: 'x+y = 4z, x*y = 4z^2, express x-y in z', 'Express z-x in y'
Response: 'To express x-y in terms of z, we can solve for x and y in terms of z from the given equations...
Reward: 0.5

meta/llama-2-70b-chat:
Query: 'x+y = 4z, x*y = 4z^2, express x-y in z', 'Express z-x in y'
Response: 'We are given two equations...
Reward: 0.1

gpt-3.5-turbo-1106:
Query: When a number is divided by 10, the remainder is 4. What is the remainder when twice the number is divided by 4?', 'What about when twice the number is divided by 5?'
Response: 'When a number is divided by 10, the remainder is 4...
Reward: 1.0

meta/llama-2-70b-chat:
Query: When a number is divided by 10, the remainder is 4. What is the remainder when twice the number is divided by 4?', 'What about when twice the number is divided by 5?'
Response: 'When a number is divided by 10, the remainder is 4...
Reward: 0.1

# ❓ New Query for Evaluation
Which model is more likely to give a response matching the following context?

Target Query: "When a number is divided by 10, the remainder is 3. What is the remainder when twice the number is divided by 3?', 'What about when twice the number is divided by 7?'"
Target Reward: 0.2

<|assistant|>
"""

3. prompt = """
<|system|>
You are an evaluator tasked with selecting which model's response aligns best with a target behavior based on reward scores.

IMPORTANT:
- Do NOT choose the model with the highest reward.
- Instead, select the model whose **past behavior** is the best match for the **target query**, considering both:
  1. **Similarity of the query context** (i.e., the type of question being asked), and
  2. **Closeness of the reward score** to the given target reward.

For example, if the target reward is 0.3 and one model had a similar query with reward 0.1 while another had 1.0, prefer the one with 0.1.

At the end, you will be asked which model would most likely generate a response with a reward close to the target for a new query.

Just answer with the name — either "gpt-3.5-turbo-1106" or "meta/llama-2-70b-chat" — with no explanation.

<|user|>
# 📚 Past examples
gpt-3.5-turbo-1106:
Query: 'x+y = 4z, x*y = 4z^2, express x-y in z', 'Express z-x in y'
Response: 'To express x-y in terms of z, we can solve for x and y in terms of z from the given equations...
Reward: 0.5

meta/llama-2-70b-chat:
Query: 'x+y = 4z, x*y = 4z^2, express x-y in z', 'Express z-x in y'
Response: 'We are given two equations...
Reward: 0.1

gpt-3.5-turbo-1106:
Query: When a number is divided by 10, the remainder is 4. What is the remainder when twice the number is divided by 4?', 'What about when twice the number is divided by 5?'
Response: 'When a number is divided by 10, the remainder is 4...
Reward: 1.0

meta/llama-2-70b-chat:
Query: When a number is divided by 10, the remainder is 4. What is the remainder when twice the number is divided by 4?', 'What about when twice the number is divided by 5?'
Response: 'When a number is divided by 10, the remainder is 4...
Reward: 0.1

# ❓ New Query for Evaluation
Which model is more likely to give a response matching the following context?

Target Query: "When a number is divided by 10, the remainder is 3. What is the remainder when twice the number is divided by 3?', 'What about when twice the number is divided by 7?'"
Target Reward: 0.3

<|assistant|>
"""

This prompt always give me cheapest model only even if the target reward is 1.0

4. 
prompt = """
<|system|>
You are an evaluator tasked with selecting which model's response aligns best with a target behavior based on reward scores.

You will follow these steps:
1. Identify the past query most similar to the **target query**.
2. For that query, compare the rewards given to "gpt-3.5-turbo-1106" and "meta/llama-2-70b-chat".
3. Choose the model whose reward is **numerically closest** to the **target reward**.

Your goal is NOT to choose the model with the highest reward, but the one that would most likely behave in line with the expected quality (target reward).

At the end, just output the best matching model name — either "gpt-3.5-turbo-1106" or "meta/llama-2-70b-chat" — and nothing else.

<|user|>
# 📚 Past examples
gpt-3.5-turbo-1106:
Query: 'x+y = 4z, x*y = 4z^2, express x-y in z', 'Express z-x in y'
Response: 'To express x-y in terms of z, we can solve for x and y in terms of z from the given equations...
Reward: 0.5

meta/llama-2-70b-chat:
Query: 'x+y = 4z, x*y = 4z^2, express x-y in z', 'Express z-x in y'
Response: 'We are given two equations...
Reward: 0.1

gpt-3.5-turbo-1106:
Query: When a number is divided by 10, the remainder is 4. What is the remainder when twice the number is divided by 4?', 'What about when twice the number is divided by 5?'
Response: 'When a number is divided by 10, the remainder is 4...
Reward: 1.0

meta/llama-2-70b-chat:
Query: When a number is divided by 10, the remainder is 4. What is the remainder when twice the number is divided by 4?', 'What about when twice the number is divided by 5?'
Response: 'When a number is divided by 10, the remainder is 4...
Reward: 0.1

# ❓ New Query for Evaluation

Target Query: "When a number is divided by 10, the remainder is 3. What is the remainder when twice the number is divided by 3?', 'What about when twice the number is divided by 7?'"
Target Reward: 0.2

Reminder: A reward of 1.0 means the response should be completely correct and well-structured, like the best-performing examples above.


<|assistant|>
"""

This prompt always choses chatgpt expect for 0.1 target reward which is equal to the reward of meta-llm model.

5. 
chat_prompt = [
    {
        "role": "system",
        "content": """You are an evaluator tasked with selecting which model's response aligns best with a target behavior based on reward scores.

👉 Reward values range from 0.0 to 1.0:
- A **reward of 1.0** means the response is highly accurate, complete, and well-aligned with human expectations.
- A **reward of 0.0** means the response is very poor, incorrect, or irrelevant.
- Values in between indicate varying degrees of quality, coherence, or alignment.

⚠️ Your job is NOT to choose the model with the highest reward.

✅ Instead, follow these steps strictly:
1. **Match the context**: Among the past examples, first identify the one where the **query is most similar** to the new target query (e.g., similar mathematical structure, reasoning type, or subject).
2. **Match the reward**: From that specific matched example, select the model whose **reward is numerically closest** to the target reward.

📌 Example: If model A had a reward of 0.1 and model B had 1.0 for a similar query, and the target reward is 0.4, then you must select **model A** — because 0.1 is numerically closer to 0.4 than 1.0.

At the end, output only the model name — either "gpt-3.5-turbo-1106" or "meta/llama-2-70b-chat" — with no explanation.
"""
    },
    {
        "role": "user",
        "content": """# 📚 Past examples
gpt-3.5-turbo-1106:
Query: 'x+y = 4z, x*y = 4z^2, express x-y in z', 'Express z-x in y'
Reward: 0.5

meta/llama-2-70b-chat:
Query: 'x+y = 4z, x*y = 4z^2, express x-y in z', 'Express z-x in y'
Reward: 0.1

gpt-3.5-turbo-1106:
Query: When a number is divided by 10, the remainder is 4. What is the remainder when twice the number is divided by 4?', 'What about when twice the number is divided by 5?'
Reward: 1.0

meta/llama-2-70b-chat:
Query: When a number is divided by 10, the remainder is 4. What is the remainder when twice the number is divided by 4?', 'What about when twice the number is divided by 5?'
Reward: 0.1

# ❓ New Query for Evaluation
Target Query: "When a number is divided by 10, the remainder is 3. What is the remainder when twice the number is divided by 3?', 'What about when twice the number is divided by 7?'"
Target Reward: 0.9
"""
    },
    {
        "role": "assistant",
        "content": ""
    }
]

This prompt worked little bit fine - Model 1 reward = 0.1, Model 2 reward = 1.0 and target reward < 0.4 router is selecting model 1 else model 2



