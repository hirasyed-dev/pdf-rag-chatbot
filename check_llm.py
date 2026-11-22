"""Quick check that your Gemini key works. Run:  python check_llm.py"""
from app import config, llm

if not config.LLM_API_KEY:
    raise SystemExit("LLM_API_KEY is empty. Add it to your .env file and save it.")

try:
    print("Models your key can use:")
    for model in llm.list_models():
        print("  -", model["name"].removeprefix("models/"))
    print("\nModel the app will use:", llm.get_model())
    print("Reply from Gemini:", llm.generate("You are a helpful assistant.", "Reply with one word: ready"))
    print("\nSUCCESS: your key and model work.")
except llm.LLMError as e:
    print("\nFAILED:", e)
