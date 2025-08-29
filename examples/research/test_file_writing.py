import os
import ssl
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
import httpx
from deepagents import create_deep_agent

# Disable SSL verification for corporate proxy environments
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

# Load environment variables from .env file in the same directory as this script
script_dir = Path(__file__).parent
env_path = script_dir / ".env"
load_dotenv(env_path)

# Configure Azure OpenAI model with SSL verification disabled for corporate proxy
http_client = httpx.Client(verify=False)

azure_model = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_MODEL_DEPLOYMENT"],
    azure_endpoint=os.environ["AZURE_AI_ENDPOINT"],
    api_key=os.environ["AZURE_AI_API_KEY"],
    api_version=os.environ["AZURE_AI_API_VERSION"],
    max_tokens=32000,
    http_client=http_client,
    temperature=0.1
)

# Simple research instructions focused on file generation
research_instructions = """
You are a research assistant. Your task is to:

1. FIRST: Use the write_file tool to write the user's question to a file called "question.txt"
2. THEN: Write a comprehensive research report to a file called "final_report.md"

You MUST use the write_file tool for both files. This is mandatory!

Example:
- write_file(file_path="question.txt", content="What is the user asking about?")
- write_file(file_path="final_report.md", content="# Research Report\n\nYour detailed research content here...")

Always write in English only.
"""

# Create the agent without subagents to test file writing
agent = create_deep_agent(
    [],  # No additional tools
    research_instructions,
    model=azure_model,
).with_config({"recursion_limit": 50})

# Test the agent
if __name__ == "__main__":
    print("File Writing Test")
    print("=" * 40)
    
    try:
        query = "What are the benefits of sustainable packaging?"
        print(f"Query: {query}")
        
        result = agent.invoke({"messages": [{"role": "user", "content": query}]})
        
        print("\nAgent execution completed!")
        
        # Check the result state for files
        print("\nChecking agent state for files...")
        if "files" in result:
            print(f"Files in agent state: {list(result['files'].keys())}")
            for filename, content in result["files"].items():
                print(f"\n--- {filename} ---")
                print(content[:300] + "..." if len(content) > 300 else content)
        else:
            print("No files found in agent state")
        
        # Show all messages
        print("\nAll messages:")
        if result and "messages" in result:
            for i, msg in enumerate(result["messages"]):
                print(f"{i+1}. {type(msg).__name__}: {str(msg)[:150]}...")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
