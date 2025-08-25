import google.generativeai as genai
import os
import numpy as np

# Configure your API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyDEGBI27sIrVG7xwBYak-RQ-lhw6PTAKNM')
genai.configure(api_key=GOOGLE_API_KEY)

# --- 1. Implement the Cosine Similarity Function ---
def cosine_similarity(vec_a, vec_b):
    """
    Calculates the cosine similarity between two vectors.
    """
    # Ensure the vectors are NumPy arrays
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)
    
    # Calculate the dot product
    dot_product = np.dot(vec_a, vec_b)
    
    # Calculate the magnitude (norm) of each vector
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    # Calculate the cosine similarity
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-8
    similarity = dot_product / ((norm_a * norm_b) + epsilon)
    
    return similarity

# --- 2. Generate Embeddings to Compare ---
embedding_model = "models/embedding-001"
texts_to_embed = [
    "The cat sat on the mat.",            # Text 0
    "A feline was resting on the rug.",   # Text 1 (Similar to 0)
    "The weather is sunny today.",        # Text 2 (Unrelated to 0 and 1)
]

print("Generating embeddings...")
embeddings = genai.embed_content(
    model=embedding_model,
    content=texts_to_embed,
    task_type="SEMANTIC_SIMILARITY"
)['embedding']

# --- 3. Calculate and Compare Similarity Scores ---
# Compare two semantically similar sentences
similarity_score_1 = cosine_similarity(embeddings[0], embeddings[1])

# Compare two unrelated sentences
similarity_score_2 = cosine_similarity(embeddings[0], embeddings[2])


print("\n--- Cosine Similarity Results ---")
print(f"Similarity between '{texts_to_embed[0]}' and '{texts_to_embed[1]}':")
print(f"Score: {similarity_score_1:.4f} (This should be high)")

print(f"\nSimilarity between '{texts_to_embed[0]}' and '{texts_to_embed[2]}':")
print(f"Score: {similarity_score_2:.4f} (This should be low)")