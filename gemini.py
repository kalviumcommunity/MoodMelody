import google.generativeai as genai
import os
import numpy as np

# Configure your API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyDEGBI27sIrVG7xwBYak-RQ-lhw6PTAKNM')
genai.configure(api_key=GOOGLE_API_KEY)

# --- 1. Implement the Euclidean Distance Function ---
def euclidean_distance(vec_a, vec_b):
    """
    Calculates the L2 (Euclidean) distance between two vectors.
    """
    # Ensure the vectors are NumPy arrays
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)
    
    # Calculate the Euclidean distance
    distance = np.linalg.norm(vec_a - vec_b)
    
    return distance

# --- 2. Generate Embeddings to Compare ---
embedding_model = "models/embedding-001"
texts_to_embed = [
    "The weather is sunny today.",        # Text 0
    "It's a bright and clear day.",       # Text 1 (Similar to 0)
    "I'm going to the movies.",           # Text 2 (Unrelated to 0 and 1)
]

print("Generating embeddings...")
embeddings = genai.embed_content(
    model=embedding_model,
    content=texts_to_embed,
    task_type="SEMANTIC_SIMILARITY"
)['embedding']

# --- 3. Calculate and Compare Distance Scores ---
# Compare two semantically similar sentences
distance_1 = euclidean_distance(embeddings[0], embeddings[1])

# Compare two unrelated sentences
distance_2 = euclidean_distance(embeddings[0], embeddings[2])


print("\n--- Euclidean Distance Results ---")
print(f"Distance between '{texts_to_embed[0]}' and '{texts_to_embed[1]}':")
print(f"Score: {distance_1:.4f} (This should be low, indicating high similarity)")

print(f"\nDistance between '{texts_to_embed[0]}' and '{texts_to_embed[2]}':")
print(f"Score: {distance_2:.4f} (This should be high, indicating low similarity)")