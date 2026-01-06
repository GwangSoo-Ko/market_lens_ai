
import sys
import os

try:
    from google.adk.agents import Agent
    from google.adk.tools import google_search, FunctionTool
    from google.adk.runners import InMemoryRunner
except ImportError:
    print("google-adk not found")
    sys.exit(1)

def my_func(x: int) -> int:
    """Returns x + 1."""
    return x + 1

print(f"Type of google_search: {type(google_search)}")
print(f"Dir of google_search: {dir(google_search)}")
print("-" * 20)
print(f"Type of FunctionTool: {type(FunctionTool)}")
print(f"Dir of FunctionTool: {dir(FunctionTool)}")

# Test 1: Mixed tools
print("\n--- Test 1: Mixed tools [google_search, my_func] ---")
try:
    agent = Agent(
        name="test_agent",
        model="gemini-1.5-flash",
        instruction="test",
        tools=[google_search, my_func]
    )
    print("Agent creation success")
except Exception as e:
    print(f"Agent creation failed: {e}")

# Test 2: Only function
print("\n--- Test 2: Only function [my_func] ---")
try:
    agent = Agent(
        name="test_agent_2",
        model="gemini-1.5-flash",
        instruction="test",
        tools=[my_func]
    )
    print("Agent creation success")
except Exception as e:
    print(f"Agent creation failed: {e}")

# Test 3: FunctionTool wrapper
print("\n--- Test 3: FunctionTool(my_func) ---")
try:
    ft = FunctionTool(my_func)
    print(f"FunctionTool created: {type(ft)}")
    agent = Agent(
        name="test_agent_3",
        model="gemini-1.5-flash",
        instruction="test",
        tools=[google_search, ft]
    )
    print("Agent creation success")
except Exception as e:
    print(f"Agent creation/Tool creation failed: {e}")

