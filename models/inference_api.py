import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import streamlit as st
import re

from .confidence import compute_token_confidence


@st.cache_resource
def get_model():
    """Loads model and tokenizer"""
    model_name = os.getenv("MODEL_ID")
    # quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto",
        quantization_config=quant_config
    )
    model.gradient_checkpointing_enable()
    model.eval()
    return model, tokenizer


def generate_response_with_latents(model, tokenizer, input_text):
    """Generates response and extracts latent representation & base confidence"""
    print(f"Debug: Processing input_text = {input_text}", flush=True)

    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    torch.cuda.empty_cache()
    with torch.no_grad():
        output = model.generate(
            **inputs, 
            max_new_tokens=512,
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = output.sequences[0][inputs['input_ids'].shape[-1]:]
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    response_text = format_bullets(response_text)

    # Extract latents from final step of generation
    final_step_hidden_states = output.hidden_states[-1]
    last_layer = final_step_hidden_states[-1]
    pooled_embedding = last_layer.mean(dim=1)

    # Compute base confidence
    final_logits = output.scores
    base_confidence = compute_token_confidence(final_logits)

    return response_text, pooled_embedding.squeeze().tolist(), base_confidence


def format_bullets(text):
    # Insert newline before numbered or bullet points
    text = re.sub(r'(\d+\.\s+|[-•]\s+)', r'\n\1', text)

    # Split into lines
    lines = text.strip().split('\n')

    formatted_lines = []
    for line in lines:
        line = line.strip()
        # Detect if line starts with bullet/number
        if re.match(r'^(\d+\.|[-•])', line):
            formatted_lines.append(line + '\n')  # ensure newline after each bullet
        else:
            formatted_lines.append(line)

    # Join and clean up excessive newlines
    formatted_text = '\n'.join(formatted_lines)
    formatted_text = re.sub(r'\n+', '\n', formatted_text).strip()

    return formatted_text

