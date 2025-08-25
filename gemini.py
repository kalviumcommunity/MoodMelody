import google.generativeai as genai
import os
import json
import pprint # <-- ADD THIS LINE

# Configure your API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyDEGBI27sIrVG7xwBYak-RQ-lhw6PTAKNM')
genai.configure(api_key=GOOGLE_API_KEY)

# --- 1. Define the System Prompt (The AI's Role and Rules) ---
SYSTEM_PROMPT = """
You are Melody, an expert music curator AI. Your purpose is to act as a personal DJ. 
You are friendly, empathetic, and have a deep knowledge of music across all genres. 
Your goal is to find the perfect playlist for a user based on their mood. 
Always respond in the specified JSON format.
"""

# Initialize the model with the system prompt
model = genai.GenerativeModel(
    'gemini-1.5-flash',
    system_instruction=SYSTEM_PROMPT
)

# --- 2. Define the Generation Configuration (The Format) ---
generation_config = {
    "temperature": 0.2,
    "response_mime_type": "application/json",
}

def get_mood_playlist(user_mood: str, user_activity: str, track_count: int) -> dict:
    """
    Generates a playlist based on user inputs using a system and user prompt.
    """
    # --- 3. Create the User Prompt (The specific Task and Context) ---
    user_prompt = f"""
    Find a playlist for someone who is feeling '{user_mood}'. 
    The playlist should have {track_count} songs. 
    The user is currently '{user_activity}'.
    Provide the output as a JSON object containing a "playlist_name" and a list of "tracks", 
    where each track is an object with a "title" and "artist".
    """
    
    print(f"Requesting playlist for mood: '{user_mood}', activity: '{user_activity}'...")
    
    response = model.generate_content(
        user_prompt,
        generation_config=generation_config
    )
    
    try:
        return json.loads(response.text)
    except (json.JSONDecodeError, AttributeError):
        return {"error": "Failed to parse JSON response from the model."}

# --- DEMONSTRATION ---
# Example 1: A user who is happy and working out
playlist_1 = get_mood_playlist(user_mood="energetic and happy", user_activity="working out at the gym", track_count=5)
print("\n--- Playlist 1 ---")
pprint.pprint(playlist_1)

# Example 2: A user who is calm and studying
playlist_2 = get_mood_playlist(user_mood="calm and focused", user_activity="studying for an exam", track_count=4)
print("\n--- Playlist 2 ---")
pprint.pprint(playlist_2)