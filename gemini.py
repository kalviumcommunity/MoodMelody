import google.generativeai as genai
import os

# It's best practice to use environment variables for API keys
# In your terminal, you might run: export GOOGLE_API_KEY='YOUR_API_KEY'
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyDEGBI27sIrVG7xwBYak-RQ-lhw6PTAKNM')
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the generative model
model = genai.GenerativeModel('gemini-1.5-flash')

def classify_email(customer_email: str) -> str:
    """
    Takes a customer email string and returns its classification using a dynamic prompt.
    """
    # This f-string is our dynamic prompt template.
    # The {customer_email} variable will be replaced with the function's input.
    dynamic_prompt = f"""
    Classify the following customer email into one of these categories: 'Billing Question', 'Technical Support', or 'General Inquiry'.

    Email: '{customer_email}'
    """
    
    print(f"Sending prompt for email: '{customer_email[:30]}...'") # Log the email being processed
    
    response = model.generate_content(dynamic_prompt)
    
    # Adding basic error handling for the response
    if response and response.text:
        return response.text.strip()
    else:
        return "Classification failed."

# --- DEMONSTRATION ---
# Now we can reuse the function for different emails.

# Email 1: A billing question
email_1 = "Hello, I was looking at my last invoice and noticed a charge I don't recognize. Can someone please help me understand what it's for? Thanks, Sarah."
classification_1 = classify_email(email_1)
print(f"-> Classification: {classification_1}\n")

# Email 2: A technical support question
email_2 = "Hi team, I can't seem to log into my account. I've tried resetting my password but the link isn't working. Can you help?"
classification_2 = classify_email(email_2)
print(f"-> Classification: {classification_2}\n")