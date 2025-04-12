from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch


# confidence = lambda * base_confidence + (1 - lambda) * retrieval_confidence
def compute_confidence_score(seq_logits, response_latents, retrieved_latents, use_rag=True):
    """Compute a composite confidence score combining base confidence and retrieval similarity (if RAG is used)"""
    # Base confidence: Composite, Entropy, Margin, Variation, Entropies
    base_confidence = compute_token_confidence(seq_logits)

    if not use_rag:
        return base_confidence['composite']
    
    factuality_score = None
    consistency_score = None

    lambda_weight = compute_dynamic_lambda(base_confidence['entropies'], retrieved_latents)
    retrieval_confidence = compute_retrieval_confidence(response_latents, retrieved_latents, consistency_score)

    # Add factuality score
    final_confidence = lambda_weight * base_confidence['composite'] + (1 - lambda_weight) * retrieval_confidence

    print(f'Composite Token Confidence: {base_confidence['composite']:.3f}')
    print(f'Entropy Confidence: {base_confidence['entropy']:.3f}')
    print(f'Margin Confidence: {base_confidence['margin']:.3f}')
    print(f'Variation Confidence: {base_confidence['variation']:.3f}\n')
    print(f'λ (dynamic): {lambda_weight:.3f}\n')
    print(f'Retrieval Confidence: {retrieval_confidence:.3f}')
    print(f'Final Confidence: {final_confidence:.3f}')
    return final_confidence


def compute_token_confidence(seq_logits):
    entropies = []
    margins = []
    variation_ratios = []

    for scores in seq_logits:
        scores = scores.squeeze(0) # remove batch dim
        probs = torch.softmax(scores, dim=-1)
        top_probs, _ = probs.topk(2, dim=-1)
        
        # Shannon Entropy: H(x) = (-sum(p(x) * log(p(x)))) for each token x
        log_probs = torch.log(probs + 1e-9)
        entropy = -torch.sum(probs * log_probs, dim=-1).item()
        entropies.append(entropy)

        # Margin of confidence: gets the difference between top-2 probs for each token
        margin = (top_probs[0] - top_probs[1]).item()
        margins.append(margin)

        # Variation ratio: how stable is top prediction (low = stable)
        variation_ratio = float(top_probs[0] < 0.5)
        variation_ratios.append(variation_ratio)

    # Normalize entropy confidence to [0,1] using max_entropy
    max_entropy = np.log(len(seq_logits[0].squeeze(0))) # Normalize by vocab size (static)
    # max_entropy = np.max(entropies) # Relative to response max entropy (dynamic)
    entropy_confidence = 1 - (np.mean(entropies) / max_entropy)   # lower entropy = higher confidence

    margin_confidence = np.mean(margins)  # larger margin = higher confidence
    variation_confidence = 1 - np.mean(variation_ratios)  # lower variation ratio = higher confidence

    # Combine (all higher -> higher confidence)
    composite_confidence = (entropy_confidence + margin_confidence + variation_confidence) / 3

    return {
        'composite': composite_confidence,
        'entropy': entropy_confidence,
        'margin': margin_confidence,
        'variation': variation_confidence,
        'entropies': entropies
    }


def compute_dynamic_lambda(entropy_scores, retrieved_latents):
    # Entropy variance: variance in uncertainty (higher = more uncertainty about certain tokens)
    entropy_variance = np.var(entropy_scores)

    # Retrieval diversity: High diversity -> lower confidence in retrieval
    retrieval_matrix = np.array(retrieved_latents)
    pairwise_sim = cosine_similarity(retrieval_matrix) # Similarity between retrieved latents (square matrix)
    diversity_score = 1 - np.mean(pairwise_sim[np.triu_indices(len(retrieval_matrix), k=1)]) # Only need upper triangle (diagonal is 1s -> k=1 to skip)

    # Combine heuristics
    lambda_raw = entropy_variance - diversity_score  # Higher variance, lower diversity -> higher lambda
    lambda_dynamic = 1 / (1 + np.exp(-lambda_raw))  # sigmoid normalization [0,1]

    # High entropy variance -> rely more on retrieval confidence (low lambda)
    # High retrieval diversity -> rely more on base confidence (high lambda)
    return lambda_dynamic


def compute_retrieval_confidence(response_latents, retrieved_latents, consistency_score):
    retrieved_matrix = np.array(retrieved_latents)
    response_vector = np.array(response_latents)

    retrieved_matrix = retrieved_matrix.reshape(len(retrieved_latents), -1)
    response_vector = response_vector.reshape(1, -1)

    similarities = cosine_similarity(response_vector, retrieved_matrix)

    # Rescale cosine similarity ([-1,1] -> [0,1])
    cosine_confidence = (np.max(similarities) + 1) / 2

    retrieval_confidence = consistency_score * 0.7 + cosine_confidence * 0.3
    return retrieval_confidence