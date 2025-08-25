import google.generativeai as genai
import os
import json

# Configure your API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyDEGBI27sIrVG7xwBYak-RQ-lhw6PTAKNM')
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the generative model
model = genai.GenerativeModel('gemini-1.5-flash')

# --- STRUCTURED OUTPUT CONFIGURATION ---
generation_config = {
    "temperature": 0.1,
    "response_mime_type": "application/json", # Enforce JSON output
}

def classify_email_structured(customer_email: str) -> dict:
    """
    Classifies an email and returns a structured JSON output.
    """
    # The prompt now explicitly asks for a JSON structure
    structured_prompt = f"""
    Analyze the following customer email and return a JSON object with two keys:
    1. "classification": One of 'Billing Question', 'Technical Support', or 'General Inquiry'.
    2. "priority": One of 'Low', 'Medium', or 'High'.

    Email: '{customer_email}'
    """
    
    print(f"Requesting structured output for email: '{customer_email[:30]}...'")
    
    # Pass the updated generation_config
    response = model.generate_content(
        structured_prompt,
        generation_config=generation_config
    )
    
    try:
        # Parse the JSON string from the model's response
        parsed_response = json.loads(response.text)
        return parsed_response
    except (json.JSONDecodeError, AttributeError):
        return {"error": "Failed to parse JSON response."}

# --- DEMONstration ---
email_1 = "Hi team, I can't log in and my password reset link is broken. This is urgent as I have a deadline."
classification_1 = classify_email_structured(email_1)
print(f"-> Parsed Output: {classification_1}")
print(f"-> Priority: {classification_1.get('priority')}\n")

email_2 = "Just wondering what your holiday hours are next week. Thanks!"
classification_2 = classify_email_structured(email_2)
print(f"-> Parsed Output: {classification_2}")
print(f"-> Priority: {classification_2.get('priority')}\n")