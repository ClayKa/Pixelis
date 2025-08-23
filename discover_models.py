import os
import google.generativeai as genai

print("--- Discovering Available Gemini Models ---")

try:
    # 1. 加载 API 密钥
    print("[1/2] Loading API key...")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("'GEMINI_API_KEY' not found in environment variables.")
    genai.configure(api_key=api_key)
    print("   -> API key loaded.")

    # 2. 列出所有支持 'generateContent' 方法的模型
    print("\n[2/2] Listing all models available for 'generateContent':")
    print("-" * 50)
    
    found_model = False
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ Model Name: {m.name}")
            print(f"   - Display Name: {m.display_name}")
            print(f"   - Description: {m.description[:100]}...")
            print("-" * 50)
            found_model = True

    if not found_model:
        print("   -> No models supporting 'generateContent' found for your API key.")

except Exception as e:
    print(f"\n❌ FAILED! An error occurred: {e}")