import os
import ssl
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
import httpx
from deepagents import create_deep_agent, SubAgent

# Disable SSL verification for corporate proxy environments
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

# Force English locale
os.environ["LANG"] = "en_US.UTF-8"
os.environ["LC_ALL"] = "en_US.UTF-8"
os.environ["LANGUAGE"] = "en"

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
    temperature=0.1  # Lower temperature for more consistent output
)

# Simple mock search function (no internet required)
def mock_search(query: str, max_results: int = 5, topic: str = "general", include_raw_content: bool = False):
    """Mock search function that returns sample data"""
    return {
        "results": [
            {
                "title": "Latest AI Developments 2024",
                "url": "https://example.com/ai-2024",
                "content": "Recent advances in AI include improved language models, better multimodal capabilities, and enhanced safety measures.",
                "score": 0.95
            },
            {
                "title": "AI Breakthroughs in Healthcare",
                "url": "https://example.com/ai-healthcare",
                "content": "AI is revolutionizing healthcare with better diagnostic tools and personalized treatment plans.",
                "score": 0.90
            }
        ]
    }

# Simple research instructions focused on file generation
research_instructions = """
=== CRITICAL LANGUAGE REQUIREMENT ===
YOU MUST WRITE EVERYTHING IN ENGLISH ONLY!
- Do NOT use Russian (Русский)
- Do NOT use Chinese (中文)
- Do NOT use any other language
- ALL responses must be in English
- ALL file contents must be in English
- If you write in any language other than English, you have FAILED
=== END LANGUAGE REQUIREMENT ===

You are an expert researcher. Your job is to conduct thorough research and write a polished report.

CRITICAL: You MUST write ALL content in ENGLISH ONLY. Never use Russian, Chinese, or any other language. Always use English.

IMPORTANT: You MUST follow these steps in order:

1. FIRST: Write the original user question to a file called `question.txt`
2. THEN: Conduct your research using the available tools
3. FINALLY: Write a comprehensive report to a file called `final_report.md`

Use the `write_file` tool to create these files. Make sure to actually write the files!

The final report should be well-structured with:
- Clear headings using ## for sections
- Detailed content based on your research
- Proper markdown formatting
- A sources section at the end

Remember: You MUST create both question.txt and final_report.md files!
"""

# Create the agent with mock search
agent = create_deep_agent(
    [mock_search],
    research_instructions,
    model=azure_model,
).with_config({"recursion_limit": 100})

# Test the agent
if __name__ == "__main__":
    print("Simple Research Agent Test")
    print("=" * 40)
    
    try:
        query = "What are the latest developments in AI? IMPORTANT: You must respond in English language only. Do not use Russian or any other language. Write everything in English."
        print(f"Query: {query}")

        # Add system message to force English
        messages = [
            {"role": "system", "content": "You are an AI assistant that responds ONLY in English. Never use Russian, Chinese, or any other language."},
            {"role": "user", "content": query}
        ]

        result = agent.invoke({"messages": messages})
        
        print("\nAgent execution completed!")
        
        # Check for files and translate if needed
        if os.path.exists("question.txt"):
            print("✅ question.txt created!")
            with open("question.txt", "r", encoding="utf-8") as f:
                print(f"Content: {f.read()}")
        else:
            print("❌ question.txt not found")

        if os.path.exists("final_report.md"):
            print("✅ final_report.md created!")
            with open("final_report.md", "r", encoding="utf-8") as f:
                content = f.read()
                print(f"Length: {len(content)} characters")
                print(f"Preview: {content[:200]}...")

                # Check if content is in Russian and translate to English
                if any(ord(char) >= 1040 and ord(char) <= 1103 for char in content[:100]):  # Cyrillic characters
                    print("⚠️  Detected Russian content. Translating to English...")

                    # Use the Azure model to translate
                    translation_prompt = f"""
                    Translate the following Russian text to English. Maintain the same structure, formatting, and markdown:

                    {content}

                    Provide ONLY the English translation, no additional comments.
                    """

                    try:
                        translation_result = azure_model.invoke([{"role": "user", "content": translation_prompt}])
                        english_content = translation_result.content

                        # Save the English version
                        with open("final_report_english.md", "w", encoding="utf-8") as f:
                            f.write(english_content)
                        print("✅ English translation saved as final_report_english.md")
                        print(f"English preview: {english_content[:200]}...")
                    except Exception as e:
                        print(f"❌ Translation failed: {e}")
        else:
            print("❌ final_report.md not found")
            
        # Show all messages
        print("\nAll messages:")
        if result and "messages" in result:
            for i, msg in enumerate(result["messages"]):
                print(f"{i+1}. {type(msg).__name__}: {str(msg)[:100]}...")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
