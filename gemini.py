import google.generativeai as genai
import os
import numpy as np

# Configure your API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyDEGBI27sIrVG7xwBYak-RQ-lhw6PTAKNM')
genai.configure(api_key=GOOGLE_API_KEY)

# --- 1. Implement the Dot Product Similarity Function ---
def dot_product_similarity(vec_a, vec_b):
    """
    Calculates the dot product similarity between two vectors.
    """
    # Ensure the vectors are NumPy arrays
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)
    
    # Calculate the dot product
    similarity = np.dot(vec_a, vec_b)
    
    return similarity

# --- 2. Generate Embeddings to Compare ---
embedding_model = "models/embedding-001"
texts_to_embed = [
    "I love the new action movie!",             # Text 0
    "That film was full of adventure.",         # Text 1 (Similar to 0)
    "I'm going to cook pasta for dinner.",      # Text 2 (Unrelated to 0 and 1)
]

print("Generating embeddings...")
embeddings = genai.embed_content(
    model=embedding_model,
    content=texts_to_embed,
    task_type="SEMANTIC_SIMILARITY"
)['embedding']

# --- 3. Calculate and Compare Similarity Scores ---
# Compare two semantically similar sentences
similarity_score_1 = dot_product_similarity(embeddings[0], embeddings[1])

# Compare two unrelated sentences
similarity_score_2 = dot_product_similarity(embeddings[0], embeddings[2])


print("\n--- Dot Product Similarity Results ---")
print(f"Similarity between '{texts_to_embed[0]}' and '{texts_to_embed[1]}':")
print(f"Score: {similarity_score_1:.4f} (This should be a high positive number)")

print(f"\nSimilarity between '{texts_to_embed[0]}' and '{texts_to_embed[2]}':")
print(f"Score: {similarity_score_2:.4f} (This should be close to zero)")