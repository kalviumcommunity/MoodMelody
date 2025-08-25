import google.generativeai as genai
import os
import json
import pprint

# Configure your API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyDEGBI27sIrVG7xwBYak-RQ-lhw6PTAKNM')
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the model
model = genai.GenerativeModel('gemini-1.5-flash')

# --- CHAIN-OF-THOUGHT PROMPT CONFIGURATION ---
generation_config = {
    "temperature": 0.0,
    "response_mime_type": "application/json",
}

def solve_word_problem(problem: str) -> dict:
    """
    Solves a word problem using Chain-of-Thought prompting.
    """
    # The prompt explicitly asks the model to break down its reasoning.
    cot_prompt = f"""
    Solve the following word problem. First, think step-by-step to explain your reasoning.
    Then, provide the final answer in a JSON object with two keys: "reasoning" and "final_answer".

    Problem: "{problem}"
    """
    
    print(f"Sending CoT prompt for problem: '{problem}'")
    
    response = model.generate_content(
        cot_prompt,
        generation_config=generation_config
    )
    
    try:
        return json.loads(response.text)
    except (json.JSONDecodeError, AttributeError):
        return {"error": "Failed to parse JSON response."}

# --- DEMONSTRATION ---
word_problem = "A coffee shop has 25 tables. 15 of them seat 4 people each, and the rest seat 2 people each. What is the maximum number of people that can be seated in the coffee shop?"
solution = solve_word_problem(word_problem)

print("\n--- Solved with Chain of Thought ---")
# We can now display the reasoning and the answer separately.
print("\nReasoning:")
print(solution.get("reasoning"))

print("\nFinal Answer:")
print(solution.get("final_answer"))