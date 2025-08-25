import google.generativeai as genai
import os
import json
import pprint

# Configure your API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyDEGBI27sIrVG7xwBYak-RQ-lhw6PTAKNM')
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the model
model = genai.GenerativeModel('gemini-1.5-flash')

# --- MULTI-SHOT PROMPT CONFIGURATION ---
generation_config = {
    "temperature": 0.0,
    "response_mime_type": "application/json",
}

def convert_text_to_json(raw_text: str) -> dict:
    """
    Converts unstructured text to a clean JSON object using a multi-shot prompt.
    """
    # The prompt includes several examples to teach the model the pattern.
    multi_shot_prompt = f"""
    Your task is to parse unstructured text and convert it into a structured JSON object.

    -- EXAMPLE 1 --
    Input: "User: John Doe, Age: 30, Location: New York"
    Output: {{"name": "John Doe", "age": 30, "city": "New York"}}

    -- EXAMPLE 2 --
    Input: "Name is Jane Smith. She is 25 and lives in London."
    Output: {{"name": "Jane Smith", "age": 25, "city": "London"}}

    -- EXAMPLE 3 --
    Input: "Age: 45, City: Tokyo, Name: Ken Tanaka"
    Output: {{"name": "Ken Tanaka", "age": 45, "city": "Tokyo"}}
    -- END EXAMPLES --

    -- TASK --
    Input: "{raw_text}"
    Output:
    """
    
    print(f"Sending multi-shot prompt for text: '{raw_text}'")
    
    response = model.generate_content(
        multi_shot_prompt,
        generation_config=generation_config
    )
    
    try:
        return json.loads(response.text)
    except (json.JSONDecodeError, AttributeError):
        return {"error": "Failed to parse JSON response."}

# --- DEMONSTRATION ---
unstructured_text = "The client is David Chen from Singapore and he is 50 years old."
details = convert_text_to_json(unstructured_text)
print("\n--- Converted JSON ---")
pprint.pprint(details)