import os
import ssl
from typing import Literal
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

from tavily import TavilyClient
from langchain.chat_models import init_chat_model
from langchain_openai import AzureChatOpenAI
import httpx

from deepagents import create_deep_agent, SubAgent

# Disable SSL verification for corporate proxy environments
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

# Load environment variables from .env file in the same directory as this script
script_dir = Path(__file__).parent
env_path = script_dir / ".env"
load_dotenv(env_path)

# Configure Tavily client with SSL verification disabled
import requests
import urllib3

# Disable SSL warnings and verification globally
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Monkey patch requests to disable SSL verification
original_request = requests.Session.request
def patched_request(self, method, url, **kwargs):
    kwargs['verify'] = False
    return original_request(self, method, url, **kwargs)
requests.Session.request = patched_request

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# Configure Azure OpenAI model with SSL verification disabled for corporate proxy
sync_http_client = httpx.Client(verify=False)
async_http_client = httpx.AsyncClient(verify=False)

azure_model = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_MODEL_DEPLOYMENT"],
    azure_endpoint=os.environ["AZURE_AI_ENDPOINT"],
    api_key=os.environ["AZURE_AI_API_KEY"],
    api_version=os.environ["AZURE_AI_API_VERSION"],
    max_tokens=32000,
    http_client=sync_http_client,
    http_async_client=async_http_client
)

# Search tool to use to do research
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    search_docs = tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )
    return search_docs

# Research subagent
research_sub_agent = {
    "name": "research-agent",
    "description": "Used to research more in depth questions.",
    "prompt": "You are a dedicated researcher. Conduct thorough research and reply with detailed findings.",
    "tools": ["internet_search"],
}

# Critique subagent
critique_sub_agent = {
    "name": "critique-agent", 
    "description": "Used to critique the final report.",
    "prompt": "You are a dedicated editor. Critique the report for accuracy and completeness. Use ls and read_file to examine files.",
}

# Simple instructions that force complete workflow
simple_instructions = f"""You are a research agent. You MUST complete this exact workflow:

1. Generate timestamp: {datetime.now().strftime("%Y%m%d_%H%M%S")}
2. Save question: write_file(file_path="{{timestamp}}_question.txt", content="[user question]")
3. Research: task(description="Research [topic]", subagent_type="research-agent") 
4. Save report: write_file(file_path="{{timestamp}}_report.md", content="[research results]")
5. Critique: task(description="Critique the report", subagent_type="critique-agent")
6. Save critique: write_file(file_path="{{timestamp}}_critique.md", content="# Critique\\n\\n[critique results]")

COMPLETE ALL 6 STEPS. Use the current timestamp: {datetime.now().strftime("%Y%m%d_%H%M%S")}
"""

# Create the agent
agent = create_deep_agent(
    [internet_search],
    simple_instructions,
    model=azure_model,
    subagents=[critique_sub_agent, research_sub_agent],
).with_config({"recursion_limit": 1000})

# Test the agent
async def main():
    print("Testing complete workflow...")
    
    query = "Research the latest developments in Mushroom-based Packaging"
    print(f"Query: {query}")
    
    result = await agent.ainvoke({"messages": [{"role": "user", "content": query}]})
    
    print("\n" + "="*50)
    print("CHECKING FILES CREATED")
    print("="*50)
    
    if "files" in result and result["files"]:
        print("✅ Files found in agent state:")
        for filename, content in result["files"].items():
            print(f"  📄 {filename} ({len(content)} characters)")
            
            # Create actual files
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  ✅ Created actual file: {filename}")
            
            # Show content preview
            if filename.endswith('_critique.md'):
                print(f"  📝 CRITIQUE PREVIEW: {content[:200]}...")
    else:
        print("❌ No files found")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
