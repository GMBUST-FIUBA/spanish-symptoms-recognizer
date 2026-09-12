from google import genai
from openai import OpenAI
import anthropic

def build_api_client(provider: str):
    if provider == "gemini":
        return genai.Client()
    elif provider == "openai":
        return OpenAI()
    elif provider == "anthropic":
        return anthropic.Anthropic()
    else:
        raise Exception(f"Proveedor de API no soportado: {provider}")
