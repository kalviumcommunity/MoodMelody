import google.generativeai as genai
import os

# Configure your API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyDEGBI27sIrVG7xwBYak-RQ-lhw6PTAKNM')
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the generative model
model = genai.GenerativeModel('gemini-1.5-flash')

# --- STOP SEQUENCE CONFIGURATION ---
# We want the model to stop generating as soon as it's about to write "4."
generation_config = {
    "temperature": 0.7,
    "stop_sequences": ["4."],
}

def generate_ideas_with_stop(topic: str) -> str:
    """
    Generates a numbered list of ideas, stopping at a specific point.
    """
    prompt = f"Brainstorm 5 creative marketing ideas for a new brand of {topic}. List them in a numbered format."
    
    print(f"Generating ideas for {topic} with a stop sequence of '4.'...")
    
    # Pass the generation_config to the API call
    response = model.generate_content(
        prompt,
        generation_config=generation_config
    )
    
    if response and response.text:
        return response.text.strip()
    else:
        return "Idea generation failed."

# --- DEMONstration ---
topic = "eco-friendly coffee cups"
ideas = generate_ideas_with_stop(topic)
print("--- Generated Ideas ---")
print(ideas)
print("--- End of Output ---")