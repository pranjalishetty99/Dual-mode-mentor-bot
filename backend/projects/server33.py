from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
import os
import base64 
import io
from dotenv import load_dotenv # ⬅️ NEW IMPORT for security

# --- Configuration ---
# ✅ Load variables from .env file securely
load_dotenv() 

# ⚠️ Load the Gemini API Key from the environment variable (read from .env)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 

app = Flask(__name__)
CORS(app) 

# --- AI Client Initialization ---
try:
    # Client initialized using the secure key from the environment
    client = genai.Client(api_key=GEMINI_API_KEY) 
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    client = None

# --- In-memory storage for chat sessions ---
room_sessions = {}

# Function to get or create a chat session for a room
def get_chat_session(room_id):
    if room_id not in room_sessions:
        print(f"Creating new chat session for Room ID: {room_id}")
        
        # System instruction is crucial for setting the AI's role and tone
        config = genai.types.GenerateContentConfig(
            system_instruction="You are an encouraging and helpful AI Study Partner for a collaborative room. Your goal is to answer questions, explain concepts, and guide group discussions concisely. When generating quizzes, use a clear bulleted or numbered list format for the questions and answers. Respond in plain text, avoid markdown for clarity in the chat box."
        )
        
        # Create a new chat session with the specified model and configuration
        room_sessions[room_id] = client.chats.create(
            model='gemini-2.5-flash',
            config=config
        )
    return room_sessions[room_id]

# Utility to create a file Part from Base64
def create_file_part(base64_data, mime_type):
    """Decodes base64 string to bytes and creates a Part for the Gemini API."""
    try:
        # The base64 data coming from JS is cleaned before it is sent here
        file_bytes = base64.b64decode(base64_data)
        return genai.types.Part.from_bytes(
            data=file_bytes,
            mime_type=mime_type
        )
    except Exception as e:
        print(f"Error creating file part: {e}")
        return None

@app.route('/ask-ai', methods=['POST'])
def ask_ai():
    if not client:
        return jsonify({"error": "AI client not initialized. Check API Key."}), 500

    data = request.get_json()
    prompt = data.get('prompt', '')
    room_id = data.get('roomId') 
    file_payload = data.get('file', None)

    if not room_id:
        return jsonify({"error": "Room ID is required."}), 400

    try:
        chat = get_chat_session(room_id)
        
        content_parts = []
        is_quiz_request = False

        # 1. Process File Payload (if present)
        if file_payload:
            file_part = create_file_part(file_payload['data'], file_payload['mimeType'])
            if not file_part:
                return jsonify({"error": "Failed to process file data."}), 500
            
            content_parts.append(file_part)
            
            # 2. AUTOMATIC ACTION CHECK
            mime_type = file_payload['mimeType']
            
            if mime_type.startswith('application/pdf') or mime_type.startswith('text/'):
                if not prompt.strip():
                    # ✅ NEW DEFAULT BEHAVIOR: Provide a summary/explanation
                    prompt = "Please read this document/file and provide a concise 3-point summary and explanation of the key content."
                
                # Check if the final prompt (either user's or default) is a quiz request
                if 'quiz' in prompt.strip().lower() or 'questions' in prompt.strip().lower():
                    # Ensure the prompt is detailed for the AI if it is a quiz request
                    if 'quiz' not in prompt.strip().lower(): 
                         prompt = f"{prompt.strip()}. Please create a short quiz with exactly 3 multiple-choice questions (A, B, C or D), followed by the correct answers at the end."

                    is_quiz_request = True
                    
        # 3. Add the User's Prompt
        if prompt:
            content_parts.append(prompt)
            
        if not content_parts:
             return jsonify({"error": "Prompt or file is required."}), 400

        # 4. Send the Request
        response = chat.send_message(content_parts)
        
        # 5. Format the Response Text for Quiz requests
        if is_quiz_request:
            ai_response_text = "✨ Quiz Generated! ✨\n\n" + response.text
        else:
            ai_response_text = response.text
        
        return jsonify({"text": ai_response_text})

    except Exception as e:
        print(f"Error calling Gemini API for room {room_id}: {e}")
        return jsonify({"error": "Failed to get response from AI."}), 500

if __name__ == '__main__':
    # The server runs on port 3000, as expected by your JavaScript client
    app.run(port=3000)