import torch
from .prompt import FACTUALITY_PROMPT, CONSISTENCY_PROMPT
import re

from .utils import StopOnToken
from transformers import StoppingCriteriaList
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
    END_ID = tokenizer.convert_tokens_to_ids("<END_CRITIC>")
    stopping = StoppingCriteriaList([StopOnToken([END_ID])])

    torch.cuda.empty_cache()
    with torch.no_grad():
        generated_ids = model.generate(
            **input, 
            stopping_criteria=stopping,
            max_new_tokens=160,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = generated_ids[:, input["input_ids"].shape[-1]:]
    response_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True).split("<END_CRITIC>")[0].strip()
    print(f'{critic_type} Critic Response:\n{response_text}')
    score = get_score(response_text)

    if score is None:
        print("[WARN] No valid score found in response:")
        print(response_text)
        return 0.0
    
    return score/100

def get_score(response):
    _SCORE_RE = re.compile(r"(Factuality|Consistency)\s*Score\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)", re.I)
    match = _SCORE_RE.search(response)
    if not match: 
        return None
    return min(float(match.group(2)), 100.0)