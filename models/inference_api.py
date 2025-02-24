import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import streamlit as st

from .confidence import compute_entropy_confidence, compute_perplexity_confidence

@st.cache_resource
def get_model():
    """Loads model and tokenizer"""
    model_name = os.getenv("MODEL_ID")
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    # quant_config = BitsAndBytesConfig(load_in_8bit=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto",
        quantization_config=quant_config, 
        output_hidden_states=True,
        return_dict_in_generate=True
    )
    model.eval()
    return model, tokenizer


def generate_response_with_latents(model, tokenizer, input_text, confidence_method="entropy"):
    """Generates response and extracts latent representation & base confidence"""
    print(f"Debug: Processing input_text = {input_text}", flush=True)

    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model.generate(
            **inputs, 
            max_new_tokens=200, 
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = output.sequences[0][inputs['input_ids'].shape[-1]:]
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # Extract latents from final step of generation
    final_step_hidden_states = output.hidden_states[-1]
    last_layer = final_step_hidden_states[-1]
    pooled_embedding = last_layer.mean(dim=1)

    # Compute base confidence
    final_logits = output.scores
    if confidence_method == "entropy":
        base_confidence = compute_entropy_confidence(final_logits)
    elif confidence_method == "perplexity":
        base_confidence = compute_perplexity_confidence(final_logits)
    else:
        raise ValueError("Invalid confidence method!")

    return response_text, pooled_embedding.squeeze().tolist(), base_confidence