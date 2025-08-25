import google.generativeai as genai
import os

# Configure your API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyDEGBI27sIrVG7xwBYak-RQ-lhw6PTAKNM')
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the generative model
model = genai.GenerativeModel('gemini-1.5-flash')

def classify_email_and_count_tokens(customer_email: str) -> str:
    """
    Counts tokens for a prompt, logs it, and then classifies the email.
    """
    dynamic_prompt = f"""
    Classify the following customer email into one of these categories: 'Billing Question', 'Technical Support', or 'General Inquiry'.

    Email: '{customer_email}'
    """
    
    # --- TOKEN COUNTING STEP ---
    # 1. Count the tokens in the prompt
    token_count = model.count_tokens(dynamic_prompt)
    # 2. Log the total token count to the console
    print(f"[INFO] Token count for this prompt: {token_count.total_tokens}")
    
    print(f"Sending prompt for email: '{customer_email[:30]}...'")
    
    # 3. Make the actual API call
    response = model.generate_content(dynamic_prompt)
    
    if response and response.text:
        return response.text.strip()
    else:
        return "Classification failed."

# --- DEMONSTRATION ---
email_1 = "Hello, I was looking at my last invoice and noticed a charge I don't recognize. Can someone please help me understand what it's for? Thanks, Sarah."
classification_1 = classify_email_and_count_tokens(email_1)
print(f"-> Classification: {classification_1}\n")

email_2 = "Hi team, I can't seem to log into my account. I've tried resetting my password but the link isn't working. Can you help?"
classification_2 = classify_email_and_count_tokens(email_2)
print(f"-> Classification: {classification_2}\n")