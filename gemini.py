import google.generativeai as genai
import os
import chromadb
import pprint

# Configure your API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyDEGBI27sIrVG7xwBYak-RQ-lhw6PTAKNM')
genai.configure(api_key=GOOGLE_API_KEY)

# --- 1. Set up the Vector Database (ChromaDB) ---
# Initialize the ChromaDB client (in-memory)
client = chromadb.Client()

# Create a collection to store our document vectors
# A collection is like a table in a traditional database.
collection = client.create_collection(name="document_collection")

print("Vector database collection created.")

# --- 2. Define Documents and Embed Them ---
# These are the documents we want our AI to be able to search through.
DOCUMENTS = [
    "The new AI model from Google is called Gemini.",
    "The capital of France is Paris, known for the Eiffel Tower.",
    "ChromaDB is an open-source vector database.",
    "The Eiffel Tower is a famous landmark in Paris.",
    "Gemini is a powerful and multimodal AI model."
]

# Use the Gemini embedding model
embedding_model = "models/embedding-001"
embeddings = genai.embed_content(
    model=embedding_model,
    content=DOCUMENTS,
    task_type="RETRIEVAL_DOCUMENT"
)['embedding']

# --- 3. Add Documents to the Database ---
# We need to provide the embeddings, the original documents, and unique IDs.
collection.add(
    embeddings=embeddings,
    documents=DOCUMENTS,
    ids=[f"doc_{i}" for i in range(len(DOCUMENTS))]
)
print(f"{len(DOCUMENTS)} documents have been added to the vector database.")

# --- 4. Query the Database ---
def query_vector_db(query: str, n_results: int = 2):
    """
    Takes a user query, embeds it, and searches the vector database.
    """
    print(f"\nSearching for documents similar to: '{query}'")
    
    # First, embed the user's query using the same model
    query_embedding = genai.embed_content(
        model=embedding_model,
        content=query,
        task_type="RETRIEVAL_QUERY"
    )['embedding']
    
    # Now, query the collection to find the most similar documents
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    return results['documents'][0]

# --- DEMONSTRATION ---
# This query is conceptually similar to documents 0 and 4.
search_results = query_vector_db("What is the latest AI from Google?")

print("\n--- Top Search Results ---")
pprint.pprint(search_results)

# This query is conceptually similar to documents 1 and 3.
search_results_2 = query_vector_db("Tell me about famous places in France.")
print("\n--- Top Search Results ---")
pprint.pprint(search_results_2)