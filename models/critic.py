import torch
from .prompt import FACTUALITY_PROMPT, CONSISTENCY_PROMPT
import re
# Models: Llama 8B, Llama 7B, Mistral 7B base

def generate_critic_score(model, tokenizer, critic_type="factuality", question=None, retrieved_info=None, generated_answer=None):
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

    input = tokenizer(prompt, return_tensors="pt").to("cuda")

    torch.cuda.empty_cache()
    with torch.no_grad():
        generated_ids = model.generate(
            **input, 
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id
        )

    response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    score = get_score(response_text)

    if score is None:
        print("[WARN] No valid score found in response:")
        print(response_text)
        return 0.0
    
    return score/100

def get_score(response):
    match = re.search(r"(Correctness|Consistency)\s*Score\s*[:\-]?\s*(\d{1,3})", response, re.IGNORECASE)
    if match:
        score = int(match.group(2))
        return min(score, 100)
    return None