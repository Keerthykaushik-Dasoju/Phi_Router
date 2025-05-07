from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

local_model_path = "/work/pi_wenlongzhao_umass_edu/25/kdasoju/phi3_5_mini_instruct"

tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(local_model_path, trust_remote_code=True)
model.to("cuda" if torch.cuda.is_available() else "cpu")

# 🧠 Chat-formatted prompt preserving original logic
# chat_prompt = [
#     {
#         "role": "system",
#         "content": """You are an evaluator tasked with selecting which model's response aligns best with a target behavior based on reward scores.

# 👉 Reward values range from 1 to 10:
# - A **reward of 1** means the response is highly accurate, complete, and well-aligned with human expectations.
# - A **reward of 10** means the response is very poor, incorrect, or irrelevant.
# - Values in between indicate varying degrees of quality, coherence, or alignment.

# ⚠️ Your job is NOT to choose the model with the highest reward.

# ✅ Instead, follow these steps strictly:
# 1. **Match the context**: Among the past examples, first identify the one where the **query is most similar** to the new target query (e.g., similar mathematical structure, reasoning type, or subject).
# 2. **Match the reward**: From that specific matched examples, select the model whose **reward is numerically closest (less absolute difference)** to the target reward.

# 📌 📌 Important note:
# Do NOT assume lower reward = better.

# Instead, calculate the absolute difference between the target reward and each model’s reward, like this:

# → |Target - Model A reward| = difference A  
# → |Target - Model B reward| = difference B

# Then choose the model with the **smaller numerical difference**.

# 🧠 Example:
# If Model A's reward = 1  
# Model B's reward = 5  
# Target reward = 4  
# Then:

# |4 - 1| = 3  
# |4 - 5| = 1

# So you must choose Model B, because 1 < 3.


# At the end, output only:
# - The selected model name
# - Absolute reward differences with both models (show which model as well)
# - One-sentence explanation (optional)
# - I want to know your understanding of the past examples with models and rewards

# Do not generate more than necessary.

# """
#     },
#     {
#         "role": "user",
#         "content": """# 📚 Past examples

# Query: 'x+y = 4z, x*y = 4z^2, express x-y in z', 'Express z-x in y'
# Model: meta/llama-2-70b-chat, Reward: 1
# Model: gpt-3.5-turbo-1106, Reward: 10


# # ❓ New Query for Evaluation
# Target Query: 'x+y = 2z, x*y = 2z^2, express x-y in z'
# Target Reward: 9
# """
#     },

# chat_prompt.append({
#     "role":"assistant",
#     "content": "The model whose reward is numerically closest to the target reward of 4 is meta/llama-2-70b-chat with a reward of 1. Although the reward is not close, it is the closest among the given options. \n Explanation: The absolute difference between the target reward (4) and the reward of meta/llama-2-70b-chat (1) is 3, which is smaller than the absolute difference between the target reward (4) and the reward of \"gpt-3.5-turbo-1106\" (5), which is 1. Therefore, \"meta/llama-2-70b-chat\" is the closest match based on the reward values. \n Model: meta/llama-2-70b-chat"
# })


# chat_prompt.append({
#     "role":"system",
#     "content":"You are an evaluator tasked with selecting which model's response aligns best with a target behavior based on reward scores."
# })

# chat_prompt.append({
#     "role":"user",
#     "content": "There are two models - model A and model B, model A has a reward of 0.1 and model B has a reward of 0.5. Now given a target reward of 0.4, can you please tell me this target reward is closer to which model?"
# })

chat_prompt = [
    {
        "role":"system",
        "content": """
            You are an evaluator tasked with classifying user queries into one of the known categories based on the structure and content of the query.

            Categories:
            - abstract2title
            - arc-challenge
            - grade-school-math

            Just output the final category label. Do not include the full query or answer choice. Do not explain your reasoning.
            """
    },
    {
        "role": "user",
        "content": """# 📚 Past examples

            Query: '['please write an title based on this abstract of article: ', 'In this study, the efficient transfer printing and stable operation of the PEDOT:PSS hole extraction layer have been researched in the perovskite solar cell, in terms of the development from an enhanced hydrophobic interface of polyurethane acrylate (PUA) mold film via perfluoropolyether. First, the energy release rate of the mold film is controlled successfully for an efficient transfer process, which was confirmed by contact angle measurements compared to the normal PUA. The transfer-printed PEDOT:PSS layer exhibits comparable smoothness, and also induces the favorable crystallinity of perovskite related to the spin-coated layer, which shows similar JSC and PCE (spin-coated compared to transfer printing: 12.85% compared to 12.33%), and improved VOC. The effects of the device electrical parameters are analyzed in detail by PL mapping, charge carrier mobility, and impedance response. Furthermore, the stability of the device with transfer-printed PEDOT:PSS achieved 90% retention for approximately 40\u202fdays, which was affected by the preserved crystallinity of perovskite, and the inhibition of the degradation of ITO from XRD and XPS analyses, respectively. Consequently, the transfer-printed hole extraction layer through the interface control between PUA and PEDOT:PSS using the improved hydrophobicity contributes to maintaining the surface morphologies and device electrical properties; this correlates with the stable operation of perovskite photovoltaics.']'
            Category:  'abstract2title'

            Query: '['please write an title based on this abstract of article: ', 'All-inorganic halide perovskites hold promise for emerging thin-film photovoltaics due to their excellent thermal stability. Unfortunately, it has been challenging to achieve high-quality films over large areas using scalable methods under realistic ambient conditions. Herein, we investigated the coupling between the fluid dynamics and the structural evolution during controlled film formation for ambient scalable fabrication of CsPbI2Br perovskite films using blade coating. We simultaneously overcame the negative influences of moisture attack and the Bénard-Marangoni instability in the drying ink and achieved an ideal sequential crystallization with changing halide composition during the film formation. As a result, we produced highly crystalline, uniform, and pinhole-free CsPbI2Br films with superior photophysical and transport properties. High-performance solar cells are fabricated to achieve power conversion efficiencies (PCEs) of 14.7% for small-aperture-area (0.03\xa0cm2) devices and 12.5% for the large-aperture-area (1.0\xa0cm2) ones, the highest PCE reported to date for large-area all-inorganic perovskite solar cells.']'
            Category:  'abstract2title'

            Query: '['please write an title based on this abstract of article: ', 'Perovskite solar cells (PSCs) have attracted great attention due to their low cost and high power conversion efficiency (PCE). However, the defects and grain boundaries in perovskite films dramatically degrade their performance. Here, we show a two-step annealing method to produce mesoporous PbI2 films for growth of continuous, pinhole-free perovskite films with large grains, followed by additional ethanol vapor annealing of perovskite films to reduce the defects and grain boundaries. The large perovskite grains dramatically suppress the carrier recombination, and consequently we obtain ZnO-nanorod-based PSCs that exhibit the best efficiency of 17.3%, with high reproducibility.']'
            Category:  'abstract2title'

            Query: '['\nThe following are examples of grade school math problems and answers:\nQuestion: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? \nAnswer: 6\n\nQuestion: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? \nAnswer: 5\n\nQuestion: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? \nAnswer: 39\n\nQuestion: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nAnswer: 8\n\nQuestion: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now? \nAnswer: 9\n\n', 'A craft store makes a third of its sales in the fabric section, a quarter of its sales in the jewelry section, and the rest in the stationery section. They made 36 sales today. How many sales were in the stationery section?']'
            Category:  'grade-school-math'

            Query: '['\nThe following are examples of grade school math problems and answers:\nQuestion: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? \nAnswer: 6\n\nQuestion: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? \nAnswer: 5\n\nQuestion: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? \nAnswer: 39\n\nQuestion: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nAnswer: 8\n\nQuestion: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now? \nAnswer: 9\n\n', 'Carmen needs $7 more to have twice the amount of money that Jethro has. Meanwhile, Patricia has $60, which is 3 times as much as Jethro. What is the sum of all their money?']'
            Category:  'grade-school-math'

            Query: '['\nThe following are examples of grade school math problems and answers:\nQuestion: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? \nAnswer: 6\n\nQuestion: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? \nAnswer: 5\n\nQuestion: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? \nAnswer: 39\n\nQuestion: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nAnswer: 8\n\nQuestion: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now? \nAnswer: 9\n\n', 'Rajesh walked 10 kilometers less than 4 times the distance that Hiro walked. Together they walked 25 kilometers. How many kilometers did Rajesh walk?']'
            Category:  'grade-school-math'

            ❓ New Query for Evaluation
            Query: '[["Generate a brief answer using only the provided claims, with no personal opinions or outside knowledge. If there is no answer based on the claims, write 'N-A'.", 'claim: Together, our findings suggested that the family environment might play an important role in the acquisition and process of the depression-related personality traits.\nclaim: These results appear consistent with the importance attached to family environment as an important influence in personality development.\nclaim: The results indicated that the family environment (cohesion, support and low conflict) was a significant predictor of psychological adaptation.\nclaim: Results suggest that by early adolescence, important aspects of family environment have consolidated into individual differences in competence and/or personality.\nclaim: These results were interpreted in light of behavioral genetic data that suggest substantial genetic influences, few common environmental influences, and large within-family environmental influences on personality development.\nquestion: what are the effects that family environment has on personality? ']'
        """   
    }
]

# Query: '['Research is being conducted on using energy from the Sun to split water molecules into hydrogen and oxygen. The oxygen will be released into the environment, and hydrogen will be used for fuel. Which statement describes how this technology will benefit the environment?\nA) The Sun\'s abundant supply of energy will be used.\nB) The properties of hydrogen will be better understood.\nC) The atmosphere will be supplied with oxygen.\nD) The availability of clean resources will increase.\n']'
chat_prompt.append(
    {
        "role": "assistant",
        "content": ""
    }
)

def convert_chat_to_prompt(prompt_list):
    formatted_prompt = ""
    for turn in prompt_list:
        formatted_prompt += f"<|{turn['role']}|>\n{turn['content']}\n"
    return formatted_prompt

formatted_prompt = convert_chat_to_prompt(chat_prompt)


# Tokenize and generate
inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200)

# Decode and print result
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
