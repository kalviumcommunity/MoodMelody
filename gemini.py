import google.generativeai as genai
import os

# Configure your API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyDEGBI27sIrVG7xwBYak-RQ-lhw6PTAKNM')
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the generative model
model = genai.GenerativeModel('gemini-1.5-flash')

# --- TEMPERATURE CONFIGURATION ---
# For a classification task, we want a focused and deterministic answer.
# A low temperature ensures the model picks the most probable category.
generation_config = {
    "temperature": 0.1, 
    "top_p": 0.8,
}

def classify_email_with_temp(customer_email: str) -> str:
    """
    Classifies an email using a dynamic prompt and a specific temperature setting.
    """
    dynamic_prompt = f"""
    Classify the following customer email into one of these categories: 'Billing Question', 'Technical Support', or 'General Inquiry'.

    Email: '{customer_email}'
    """
    
    print(f"Sending prompt with temperature={generation_config['temperature']}...")
    
    # Pass the generation_config to the API call
    response = model.generate_content(
        dynamic_prompt,
        generation_config=generation_config
    )
    
    if response and response.text:
        return response.text.strip()
    else:
        return "Classification failed."

# --- DEMONstration ---
email_1 = "Hello, my screen is frozen and I can't click anything. I've already tried restarting my computer."
classification_1 = classify_email_with_temp(email_1)
print(f"-> Classification: {classification_1}\n")