import google.generativeai as genai
import os
import json
import pprint

# Configure your API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyDEGBI27sIrVG7xwBYak-RQ-lhw6PTAKNM')
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the model
model = genai.GenerativeModel('gemini-1.5-flash')

# --- ONE-SHOT PROMPT CONFIGURATION ---
generation_config = {
    "temperature": 0.0,
    "response_mime_type": "application/json",
}

def extract_details_with_one_shot(sentence: str) -> dict:
    """
    Extracts details from a sentence using a one-shot prompt.
    """
    # The prompt includes one clear example of input and desired output.
    one_shot_prompt = f"""
    Your task is to extract the product name and the sentiment from a customer review. Provide the output in JSON format.

    -- EXAMPLE --
    Input: "I absolutely love my new UltraWidget Max, it works perfectly!"
    Output: {{"product": "UltraWidget Max", "sentiment": "Positive"}}
    -- END EXAMPLE --

    -- TASK --
    Input: "{sentence}"
    Output:
    """
    
    print(f"Sending one-shot prompt for sentence: '{sentence}'")
    
    response = model.generate_content(
        one_shot_prompt,
        generation_config=generation_config
    )
    
    try:
        return json.loads(response.text)
    except (json.JSONDecodeError, AttributeError):
        return {"error": "Failed to parse JSON response."}

# --- DEMONSTRATION ---
review = "The new Pixel phone has a great camera, but the battery life is a bit disappointing."
details = extract_details_with_one_shot(review)
print("\n--- Extracted Details ---")
pprint.pprint(details)