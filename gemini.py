import google.generativeai as genai
import os
import pprint # Used for pretty-printing the output

# Configure your API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyDEGBI27sIrVG7xwBYak-RQ-lhw6PTAKNM')
genai.configure(api_key=GOOGLE_API_KEY)

# --- EMBEDDING GENERATION ---

# List of texts we want to convert into numerical vectors
texts_to_embed = [
    "What is the weather like today?",
    "How is the forecast for this afternoon?",
    "What is the capital of France?",
]

print("Generating embeddings for the following texts:")
for text in texts_to_embed:
    print(f"- {text}")

# Call the embedding model
# Note: We use a specific model name for embeddings.
result = genai.embed_content(
    model="models/embedding-001",
    content=texts_to_embed,
    task_type="RETRIEVAL_DOCUMENT" # Specifies the intended use case
)

# The result contains a list of embeddings, one for each text.
for i, embedding in enumerate(result['embedding']):
    print(f"\n--- Embedding for text #{i+1} ---")
    # A real embedding is a long list of numbers. We'll just show the first few.
    print(f"Vector (first 5 numbers): {embedding[:5]}")
    print(f"Vector dimension: {len(embedding)}")