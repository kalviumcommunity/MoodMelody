import google.generativeai as genai
import os

# Configure your API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyDEGBI27sIrVG7xwBYak-RQ-lhw6PTAKNM')
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the generative model
model = genai.GenerativeModel('gemini-1.5-flash')

# --- TOP P CONFIGURATION ---
# We create a configuration object to control the model's generation process.
# For a classification task, we want a focused and less random answer,
# so a lower top_p is suitable.
generation_config = {
    "temperature": 0.2,
    "top_p": 0.8, 
}

def classify_email_with_topp(customer_email: str) -> str:
    """
    Classifies an email using a dynamic prompt and a specific top_p setting.
    """
    dynamic_prompt = f"""
    Classify the following customer email into one of these categories: 'Billing Question', 'Technical Support', or 'General Inquiry'.

    Email: '{customer_email}'
    """
    
    print(f"Sending prompt with top_p={generation_config['top_p']}...")
    
    # Pass the generation_config to the API call
    response = model.generate_content(
        dynamic_prompt,
        generation_config=generation_config
    )
    
    if response and response.text:
        return response.text.strip()
    else:
        return "Classification failed."

# --- DEMONSTRATION ---
email_1 = "Hi team, I can't seem to log into my account. I've tried resetting my password but the link isn't working. Can you help?"
classification_1 = classify_email_with_topp(email_1)
print(f"-> Classification: {classification_1}\n")