import os
import google.generativeai as genai
import time

print("--- Gemini Connection Smoke Test ---")

try:
    # 1. 加载 API 密钥
    print("[1/3] Loading API key from environment variable 'GEMINI_API_KEY'...")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("   -> FATAL ERROR: 'GEMINI_API_KEY' not found in environment variables.")
        exit()
    genai.configure(api_key=api_key)
    print("   -> API key loaded and configured.")

    # 2. 初始化一个最简单的文本模型
    print("[2/3] Initializing a text-only model (gemini-pro)...")
    model = genai.GenerativeModel('gemini-pro')
    print("   -> Model initialized.")

    # 3. 发送一个最简单的文本请求
    prompt = "Hello, world. If you can read this, respond with 'OK'."
    print(f"[3/3] Sending a simple text request: '{prompt}'")
    print("   -> Waiting for response...")
    
    start_time = time.time()
    response = model.generate_content(prompt)
    end_time = time.time()
    
    print("\n--- TEST RESULT ---")
    print(f"✅ SUCCESS! Received a response in {end_time - start_time:.2f} seconds.")
    print(f"Response text: {response.text}")

except Exception as e:
    print("\n--- TEST RESULT ---")
    print(f"❌ FAILED! An error occurred: {e}")