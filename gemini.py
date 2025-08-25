import google.generativeai as genai
import os

# Configure your API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyDEGBI27sIrVG7xwBYak-RQ-lhw6PTAKNM')
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the generative model
model = genai.GenerativeModel('gemini-1.5-flash')

# --- TOP K CONFIGURATION ---
# For a classification task, we want a highly predictable answer.
# Setting a low top_k ensures the model only considers the most
# likely categories and doesn't get creative.
generation_config = {
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 5, 
}

def classify_email_with_topk(customer_email: str) -> str:
    """
    Classifies an email using a dynamic prompt and a specific top_k setting.
    """
    dynamic_prompt = f"""
    Classify the following customer email into one of these categories: 'Billing Question', 'Technical Support', or 'General Inquiry'.

    Email: '{customer_email}'
    """
    
    print(f"Sending prompt with top_k={generation_config['top_k']}...")
    
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
email_1 = "I was wondering what your business hours are for this upcoming holiday."
classification_1 = classify_email_with_topk(email_1)
print(f"-> Classification: {classification_1}\n")