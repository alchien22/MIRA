from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

# confidence = lambda * base_confidence + (1 - lambda) * retrieval_confidence

def compute_confidence_score(response_latents, retrieved_latents, base_confidence, use_rag):
    """Compute a composite confidence score combining base confidence and retrieval similarity (if RAG is used)"""
    if not use_rag:
        return base_confidence

    # Compute retrieval confidence
    retrieved_matrix = np.array(retrieved_latents)
    response_vector =  np.array(response_latents).reshape(1, -1)
    similarities = cosine_similarity(response_vector, retrieved_matrix)

    # Rescale cosine similarity ([-1,1] -> [0,1])
    retrieval_confidence = (np.max(similarities) + 1) / 2

    # lambda_weight = compute_dynamic_lambda(retrieved_latents, response_latents)
    lambda_weight = 0.5

    # Composite confidence score
    final_confidence = lambda_weight * base_confidence + (1 - lambda_weight) * retrieval_confidence
    return final_confidence


def compute_entropy_confidence(logits, max_entropy=5):
    """Computes base confidence based on entropy (lower = higher confidence)"""
    probs = torch.nn.functional.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-9)
    entropy = -torch.sum(probs * log_probs, dim=-1).mean().item()
    base_confidence = 1 - (entropy / max_entropy)
    return base_confidence


def compute_perplexity_confidence(logits):
    """Computes base confidence using perplexity (lower = higher confidence)"""
    probs = torch.nn.functional.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-9)
    perplexity = torch.exp(-log_probs.mean()).item()
    base_confidence = 1 / (1 + perplexity)
    return max(0.0, min(1.0, base_confidence))


def compute_dynamic_lambda(retrieved_latents, response_latents):
    """Dynamically adjust lambda based on retrieval quality."""
    if not retrieved_latents:
        return 1.0
    retrieved_matrix = np.array(retrieved_latents)
    response_vector = np.array(response_latents).reshape(1, -1)
    similarities = cosine_similarity(response_vector, retrieved_matrix)

    retrieval_strength = np.mean(similarities)  # Mean similarity score
    lambda_dynamic = 1 / (1 + retrieval_strength)  # Inverse scaling

    return max(0.2, min(0.8, lambda_dynamic))  # Keep λ within [0.2, 0.8]


def batch_extract_latents(model, tokenizer, texts, confidence_method="entropy"):
    """Batch processes retrieved documents to extract latent representations efficiently."""
    if not texts:
        return []

    if isinstance(texts, str):
        texts = [texts]

    # Tokenize all texts together (batch processing)
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        # Forward pass through the model (without generating output tokens)
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    # Extract last hidden layer
    last_hidden_states = outputs.hidden_states[-1]  # Shape: (batch_size, seq_len, hidden_dim)

    if len(texts) == 1:
        last_hidden_states = last_hidden_states.unsqueeze(0)  # Add batch dimension if only 1 string

    # Mean pooling over token embeddings for each document
    pooled_embeddings = last_hidden_states.mean(dim=1)  # Shape: (batch_size, hidden_dim)

    return pooled_embeddings.tolist()
