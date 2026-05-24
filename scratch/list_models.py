import os
from google import genai
from app.config import get_settings

def list_models():
    settings = get_settings()
    api_key = settings.google_api_key
    if not api_key:
        print("Error: GOOGLE_API_KEY is not configured in .env")
        return
        
    print(f"Initializing genai Client...")
    client = genai.Client(api_key=api_key)
    
    try:
        print("Listing models...")
        # In the new google-genai SDK, models are listed via client.models.list()
        for m in client.models.list():
            print(f"Model: {m.name}, Display: {m.display_name}, Supported Actions: {m.supported_actions}")
    except Exception as e:
        print(f"Failed to list models: {e}")

if __name__ == "__main__":
    list_models()
