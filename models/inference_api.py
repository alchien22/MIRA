import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .confidence import compute_entropy_confidence, compute_perplexity_confidence

def get_model():
    """Loads model and tokenizer"""
    model_name = os.getenv("MODEL_ID")
    quant_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        quantization_config=quant_config, 
        output_hidden_states=True
    )
    
    model.eval()
    return model, tokenizer


def generate_response_with_latents(model, tokenizer, input_text, confidence_method="entropy"):
    """Generates response and extracts latent representation & base confidence"""
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    # Extract last hidden layer
    last_hidden_states = outputs.hidden_states[-1]
    # Mean pool to get sentence-level representation (don't want latent vectors for each token)
    pooled_embedding = last_hidden_states.mean(dim=1)

    # Compute base confidence
    logits = outputs.logits
    if confidence_method == "entropy":
        base_confidence = compute_entropy_confidence(logits)
    elif confidence_method == "perplexity":
        base_confidence = compute_perplexity_confidence(logits)
    else:
        raise ValueError("Invalid confidence method!")

    # Generate response
    generated_tokens = model.generate(**inputs, max_new_tokens=100)
    response_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    return response_text, pooled_embedding.squeeze().tolist(), base_confidence