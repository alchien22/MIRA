from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

def compute_confidence_score(base_confidence, response_latents, retrieved_latents, use_rag=True, factuality_score=None, consistency_score=None):
    """
        Compute a composite confidence score combining factuality, model confidence, and retrieval confidence (weighted by a dynamic lambda)
            Factuality score: from a critic model
            Model confidence: token-level confidence (entropy, margin, variation)
            Retrieval confidence: based on cosine similarity and consistency score
            Dyamic lambda: weight based on variation in response entropies and retrieval diversity
    """
    if not use_rag:
        return base_confidence['composite']

    lambda_weight = 0.5 #compute_dynamic_lambda(base_confidence['entropies'], retrieved_latents)
    retrieval_confidence = compute_retrieval_confidence(response_latents, retrieved_latents, consistency_score)

    # Confidence: factuality * (lambda * model_confidence + (1 - lambda) * retrieval_confidence)
    final_confidence = factuality_score * (lambda_weight * base_confidence['composite'] + (1 - lambda_weight) * retrieval_confidence)

    print(f'Factuality Score: {factuality_score:.3f}')
    # print(f'λ (dynamic): {lambda_weight:.3f}\n')
    print(f"Model Confidence: {base_confidence['composite']:.3f}")
    # print(f'Entropy Confidence: {base_confidence['entropy']:.3f}')
    # print(f'Margin Confidence: {base_confidence['margin']:.3f}')
    # print(f'Variation Confidence: {base_confidence['variation']:.3f}\n')
    print(f'Retrieval Confidence: {retrieval_confidence:.3f}')
    # print(f'Final Confidence: {final_confidence:.3f}')
    return final_confidence


def compute_token_confidence(seq_logits):
    '''
        composite token confidence: avg of entropy, margin, and variation confidence
            entropy confidence: 1 - avg_entropy / max_entropy (entropy meaning shannon entropy of the token probs over entire vocab)
            margin confidence: avg. difference between top-2 probs in the vocab. for each token
            variation confidence: 1 - avg_variation_ratio (where variation ratio is 0 if top-1 prob > 0.5)
    '''
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


def compute_retrieval_confidence(response_latents, retrieved_latents, consistency_score):
    '''
        retrieval confidence: weighted sum of cosine similarity and consistency score from a critic model
    '''
    response_vector = np.atleast_2d(response_latents)  # shape (1, D)
    retrieved_matrix = np.atleast_2d(retrieved_latents) # shape (N, D)

    similarities = cosine_similarity(response_vector, retrieved_matrix)

    # Rescale cosine similarity ([-1,1] -> [0,1])
    cosine_confidence = (np.max(similarities) + 1) / 2

    # Weight consistency score more heavily (from a critic model)
    weight = 0.7
    retrieval_confidence = weight * consistency_score + (1 - weight) * cosine_confidence
    print(f'cosine sim: {cosine_confidence}')
    return retrieval_confidence


def compute_dynamic_lambda(entropy_scores, retrieved_latents):
    '''
        dynamic lambda: weight decreases with entropy variance and increases with higher retrieval diversity
            (higher variance = lower model confidence)
            (diversity = likely worse retrievals)
    '''
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