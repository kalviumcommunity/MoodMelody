import google.generativeai as genai
import os
import json

# Configure your API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyDEGBI27sIrVG7xwBYak-RQ-lhw6PTAKNM')
genai.configure(api_key=GOOGLE_API_KEY)

# --- 1. Define the Python function (our "tool") ---
def get_weather(location: str, unit: str = "celsius"):
    """
    A mock function to get the current weather for a given location.
    In a real app, this would call a weather API.
    """
    print(f"--- Calling get_weather(location='{location}', unit='{unit}') ---")
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "15", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "20", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})

# --- 2. Describe the tool to the AI model ---
model = genai.GenerativeModel(
    'gemini-1.5-flash',
    tools=[get_weather]
)

# --- 3. Send a prompt that requires the tool ---
chat = model.start_chat()
prompt = "What is the current temperature in San Francisco?"
print(f"User Prompt: {prompt}")

response = chat.send_message(prompt)

# --- 4. Interpret the model's response ---
function_call = response.candidates[0].content.parts[0].function_call
print(f"Model wants to call function: {function_call.name}")

# --- THE FIX IS HERE ---
# Check if function_call.args exists before trying to convert it to a dict.
args = dict(function_call.args) if function_call.args else {}
print(f"With arguments: {args}")

# --- 5. Execute the function ---
if function_call.name == "get_weather":
    weather_data = get_weather(location=args.get("location"), unit=args.get("unit"))
    
    print("--- Function Executed. Result: ---")
    print(weather_data)