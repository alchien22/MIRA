import torch
from .prompt import FACTUALITY_PROMPT, CONSISTENCY_PROMPT
import re
from openai import OpenAI
import google.generativeai as genai
import os

from .utils import StopOnToken
from transformers import StoppingCriteriaList
# Models: Llama 8B, Llama 7B, Mistral 7B base


def generate_critic_score(model, tokenizer, critic_type="factuality", question=None, retrieved_info=None, generated_answer=None, model_type='local'):
    if critic_type == "factuality":
        prompt = FACTUALITY_PROMPT.format(
            question=question,
            retrieved_info=retrieved_info,
            generated_answer=generated_answer
        )
    elif critic_type == "consistency":
        prompt = CONSISTENCY_PROMPT.format(
            retrieved_info=retrieved_info,
            generated_answer=generated_answer
        )

    if model_type == 'local':
        input = tokenizer(prompt, return_tensors="pt").to("cuda")

        END_ID = tokenizer.convert_tokens_to_ids("<END_CRITIC>")
        stopping = StoppingCriteriaList([StopOnToken([END_ID])])

        torch.cuda.empty_cache()
        with torch.no_grad():
            generated_ids = model.generate(
                **input, 
                stopping_criteria=stopping,
                max_new_tokens=512,
                pad_token_id=tokenizer.eos_token_id
            )
        # print("--- RAW OUTPUT ---")
        # print(tokenizer.decode(generated_ids[0], skip_special_tokens=False))

        generated_tokens = generated_ids[:, input["input_ids"].shape[-1]:]
        response_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True).split("<END_CRITIC>")[0].strip()

    elif model_type == 'gpt':
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response_text = call_model(client, prompt, max_tokens=512, temperature=0.7, model='gpt')

    elif model_type == 'gemini':
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        client = genai.GenerativeModel(model_name="gemini-2.0-flash")
        response_text = call_model(client, prompt, max_tokens=512, temperature=0.7, model='gemini')

    print(f'{critic_type} Critic Response:\n{response_text}')
    score = get_score(response_text)

    if score is None:
        print("[WARN] No valid score found in response:")
        print(response_text)
        return 0.0
    
    return score/100


def call_model(client, prompt, max_tokens=400, temperature=0.7, model='gpt'):
    if model == 'gpt':
        model_id = "gpt-4.1-mini"
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    elif model == 'gemini':
        gemini_response = client.generate_content(
            contents=[
                {
                    "role": "user", 
                    "parts": [prompt]
                }
            ],
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature
            }
        )
        return gemini_response.text


def get_score(response):
    match = re.search(r"(Factuality|Consistency)\s*Score\s*[:\-]?\s*(\d{1,3})", response, re.IGNORECASE)
    if match:
        return min(int(match.group(2)), 100)
    return None