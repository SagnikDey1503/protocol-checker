import os
from app.config import get_settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def test():
    settings = get_settings()
    api_key = settings.google_api_key
    if not api_key:
        print("Error: GOOGLE_API_KEY is not configured in .env")
        return
        
    model_name = "models/gemini-embedding-2"
    print(f"Testing model: {model_name}...")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=api_key
        )
        vec = embeddings.embed_query("Hello world")
        print(f"SUCCESS! Default Dimension: {len(vec)}")
        
        # Test custom output_dimensionality
        try:
            embeddings_custom = GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=api_key,
                output_dimensionality=384
            )
            vec_custom = embeddings_custom.embed_query("Hello world")
            print(f"SUCCESS with dimension 384! Custom dimension: {len(vec_custom)}")
        except Exception as e:
            print(f"Could not use output_dimensionality=384: {e}")
            
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test()
