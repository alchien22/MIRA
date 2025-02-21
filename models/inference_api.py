import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from confidence import compute_entropy_confidence, compute_perplexity_confidence
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# def get_model():
      
# 	llm = HuggingFaceEndpoint(
# 		repo_id=os.getenv("MODEL_ID"),
# 		task="text-generation",
# 		max_new_tokens=100,
# 		do_sample=False
# 	)

# 	llm_chat_engine = ChatHuggingFace(llm=llm)
# 	return llm_chat_engine


def get_model():
    """Loads model and tokenizer"""
    model_name = os.getenv("MODEL_ID")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
    
    model.eval()

    return model, tokenizer


def generate_response_with_latents(model, tokenizer, input_text, confidence_method="entropy"):
    """Generates response and extracts latent representation & base confidence"""
    inputs = tokenizer(input_text, return_tensors="pt")

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