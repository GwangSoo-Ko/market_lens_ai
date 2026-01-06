import asyncio
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 프로젝트 루트를 경로에 추가
sys.path.append(os.getcwd())

from adk_utils import AdkAgentRunner, _ensure_google_api_key
try:
    from google.adk.agents import Agent
    from google.genai import types
except ImportError:
    print("google-adk not installed")
    sys.exit(0)

async def test_runner():
    # API 키 확인
    try:
        _ensure_google_api_key()
    except ValueError:
        print("API Key missing, skipping test")
        return

    print("--- Test Start ---")
    agent = Agent(name="test_agent", model="gemini-1.5-flash", instruction="You are a helpful assistant.")
    runner = AdkAgentRunner(agent)
    
    # 텍스트 + 더미 파일 파트 테스트
    prompt = "Hello, this is a test."
    # types.Part(text=...) 사용
    parts = [types.Part(text=prompt)]
    new_message = types.Content(role="user", parts=parts)
    
    print(f"Calling run_text_async with new_message type: {type(new_message)}")
    try:
        res = await runner.run_text_async(new_message=new_message)
        print("Success Result:", res)
    except Exception as e:
        print("Error in test:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_runner())

