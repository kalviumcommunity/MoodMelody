import google.generativeai as genai

# Configure your API key (replace with your actual key)
# It's best practice to store keys securely, not directly in code.
GOOGLE_API_KEY = 'AIzaSyDEGBI27sIrVG7xwBYak-RQ-lhw6PTAKNM'
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the generative model with a current name
model = genai.GenerativeModel('gemini-1.5-flash') 

# --- This is our Zero-Shot Prompt ---
zero_shot_prompt = """
Classify the following customer email into one of these categories: 'Billing Question', 'Technical Support', or 'General Inquiry'.

Email: 'Hello, I was looking at my last invoice and noticed a charge I don't recognize. Can someone please help me understand what it's for? Thanks, Sarah.'
"""

print("Sending prompt to the model...")

# Send the prompt to the model and get the response
response = model.generate_content(zero_shot_prompt)

print("---")
# The model's output will be the classification.
# .text is used to get the clean text part of the response.
print(f"Model's Classification: {response.text}")
print("---")