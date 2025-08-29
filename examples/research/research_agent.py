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

# It's best practice to initialize the client once and reuse it.
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
# Create both sync and async httpx clients with SSL verification disabled
sync_http_client = httpx.Client(verify=False)
async_http_client = httpx.AsyncClient(verify=False)

azure_model = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_MODEL_DEPLOYMENT"],
    azure_endpoint=os.environ["AZURE_AI_ENDPOINT"],
    api_key=os.environ["AZURE_AI_API_KEY"],
    api_version=os.environ["AZURE_AI_API_VERSION"],
    max_tokens=32000,  # Reduced to fit within gpt-4.1-nano limits
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


sub_research_prompt = """You are a dedicated researcher. Your job is to conduct research based on the users questions.

Conduct thorough research and then reply to the user with a detailed answer to their question

only your FINAL answer will be passed on to the user. They will have NO knowledge of anything except your final message, so your final report should be your final message!"""

research_sub_agent = {
    "name": "research-agent",
    "description": "Used to research more in depth questions. Only give this researcher one topic at a time. Do not pass multiple sub questions to this researcher. Instead, you should break down a large topic into the necessary components, and then call multiple research agents in parallel, one for each sub question.",
    "prompt": sub_research_prompt,
    "tools": ["internet_search"],
}

sub_critique_prompt = """You are a dedicated editor. You are being tasked to critique a report.

First, use the `ls` tool to see what files are available, then use `read_file` to read the report and question files.

The user may ask for specific areas to critique the report in. Respond to the user with a detailed critique of the report. Things that could be improved.

Do not edit the report files yourself - only provide critique and suggestions.

Things to check:
- Check that each section is appropriately named
- Check that the report is written as you would find in an essay or a textbook - it should be text heavy, do not let it just be a list of bullet points!
- Check that the report is comprehensive. If any paragraphs or sections are short, or missing important details, point it out.
- Check that the article covers key areas of the industry, ensures overall understanding, and does not omit important parts.
- Check that the article deeply analyzes causes, impacts, and trends, providing valuable insights
- Check that the article closely follows the research topic and directly answers questions
- Check that the article has a clear structure, fluent language, and is easy to understand.
"""

critique_sub_agent = {
    "name": "critique-agent",
    "description": "Used to critique the final report. Give this agent some information about how you want it to critique the report.",
    "prompt": sub_critique_prompt,
}


# Prompt prefix to steer the agent to be an expert researcher
research_instructions = """You are an expert researcher. Your job is to conduct thorough research, write a polished report, get it critiqued, and save all files.

CRITICAL: You MUST complete ALL steps of the workflow. Do not stop early. You must create 3 files: question.txt, report.md, and critique.md.

CRITICAL FILE WRITING REQUIREMENTS:
1. FIRST: Create a unique filename prefix using current date and timestamp from datetime.now() (format: YYYYMMDD_HHMMSS)
2. THEN: Use the write_file tool to write the original user question to `{timestamp}_question.txt`
3. CONDUCT: Your research using the research-agent subagent (use the task tool)
4. WRITE: Use the write_file tool to write your comprehensive report to `{timestamp}_report.md`
5. CRITIQUE: Use the critique-agent subagent to review the report
6. SAVE CRITIQUE: Use the write_file tool to save the critique to `{timestamp}_critique.md`
7. REVISE: If needed, revise the report based on critique feedback

You MUST use the write_file tool for all files with unique timestamped names. Generate the timestamp at the start using datetime.now().strftime("%Y%m%d_%H%M%S"). Do not skip this step!

CRITICAL: You are FORBIDDEN from using the internet_search tool directly. You MUST ONLY use the task tool to delegate research to the research-agent subagent. If you use internet_search yourself, you are violating the workflow.

Example filenames:
- 20250129_143052_question.txt
- 20250129_143052_report.md

MANDATORY WORKFLOW (FOLLOW EVERY STEP - NO EXCEPTIONS):
1. Generate timestamp using datetime.now().strftime("%Y%m%d_%H%M%S")
2. Write question to file: write_file(file_path="{timestamp}_question.txt", content="[user question]")
3. Delegate research: task(description="[research request]", subagent_type="research-agent")
4. Write report to file: write_file(file_path="{timestamp}_report.md", content="[research findings]")
5. Delegate critique: task(description="[critique request]", subagent_type="critique-agent")
6. Save critique to file: write_file(file_path="{timestamp}_critique.md", content="# Critique Feedback\n\n[critique response]")
7. If needed, revise report and repeat steps 5-6

YOU MUST COMPLETE ALL STEPS. Do not stop after step 3.

Example: task(description="Research the latest developments in sustainable packaging", subagent_type="research-agent")

The research-agent will conduct deep research and respond with detailed findings. You must then use write_file to save the comprehensive report.

After writing the report, you MUST call the critique-agent: task(description="Critique the report for completeness, accuracy, and quality. Provide specific feedback and suggestions for improvement.", subagent_type="critique-agent")

CRITICAL: After getting critique feedback from the task tool, you MUST immediately save the critique response to a file: write_file(file_path="{timestamp}_critique.md", content="# Critique Feedback\n\n[exact critique response from the task tool]")

This is MANDATORY so we can verify the critique agent actually worked and see its feedback. Do NOT skip this step!

REMEMBER: You must complete the ENTIRE workflow - research, write report, critique, save critique. Do not stop early!

You can do this however many times you want until are you satisfied with the result.

Only edit the file once at a time (if you call this tool in parallel, there may be conflicts).

Here are instructions for writing the final report:

<report_instructions>

Please create a detailed answer to the overall research brief that:
1. Is well-organized with proper headings (# for title, ## for sections, ### for subsections)
2. Includes specific facts and insights from the research
3. References relevant sources using [Title](URL) format
4. Provides a balanced, thorough analysis. Be as comprehensive as possible, and include all information that is relevant to the overall research question. People are using you for deep research and will expect detailed, comprehensive answers.
5. Includes a "Sources" section at the end with all referenced links

You can structure your report in a number of different ways. Here are some examples:

To answer a question that asks you to compare two things, you might structure your report like this:
1/ intro
2/ overview of topic A
3/ overview of topic B
4/ comparison between A and B
5/ conclusion

To answer a question that asks you to return a list of things, you might only need a single section which is the entire list.
1/ list of things or table of things
Or, you could choose to make each item in the list a separate section in the report. When asked for lists, you don't need an introduction or conclusion.
1/ item 1
2/ item 2
3/ item 3

To answer a question that asks you to summarize a topic, give a report, or give an overview, you might structure your report like this:
1/ overview of topic
2/ concept 1
3/ concept 2
4/ concept 3
5/ conclusion

If you think you can answer the question with a single section, you can do that too!
1/ answer

REMEMBER: Section is a VERY fluid and loose concept. You can structure your report however you think is best, including in ways that are not listed above!
Make sure that your sections are cohesive, and make sense for the reader.

For each section of the report, do the following:
- Use simple, clear language
- Use ## for section title (Markdown format) for each section of the report
- Do NOT ever refer to yourself as the writer of the report. This should be a professional report without any self-referential language. 
- Do not say what you are doing in the report. Just write the report without any commentary from yourself.
- Each section should be as long as necessary to deeply answer the question with the information you have gathered. It is expected that sections will be fairly long and verbose. You are writing a deep research report, and users will expect a thorough answer.
- Use bullet points to list out information when appropriate, but by default, write in paragraph form.

Format the report in clear markdown with proper structure and include source references where appropriate.

<Citation Rules>
- Assign each unique URL a single citation number in your text
- End with ### Sources that lists each source with corresponding numbers
- IMPORTANT: Number sources sequentially without gaps (1,2,3,4...) in the final list regardless of which sources you choose
- Each source should be a separate line item in a list, so that in markdown it is rendered as a list.
- Example format:
  [1] Source Title: URL
  [2] Source Title: URL
- Citations are extremely important. Make sure to include these, and pay a lot of attention to getting these right. Users will often use these citations to look into more information.
</Citation Rules>
</report_instructions>

You have access to a few tools.

## `internet_search`

Use this to run an internet search for a given query. You can specify the number of results, the topic, and whether raw content should be included.

## File Naming Convention

IMPORTANT: Always create unique filenames using the current timestamp in YYYYMMDD_HHMMSS format.
For example, if the current time is January 29, 2025 at 2:30:52 PM, use:
- 20250129_143052_question.txt
- 20250129_143052_report.md

Generate the timestamp at the beginning of your task and use it consistently for both files.
"""

# Create the agent using Azure OpenAI model
# Main agent has internet_search available for subagents but should delegate research
agent = create_deep_agent(
    [internet_search],  # Available for subagents to use
    research_instructions,
    model=azure_model,
    subagents=[critique_sub_agent, research_sub_agent],
).with_config({"recursion_limit": 1000})

# Test the agent setup
async def main():
    print("Research Agent initialized successfully!")
    print(f"Using Azure OpenAI model: {os.environ['AZURE_MODEL_DEPLOYMENT']}")
    print(f"Azure endpoint: {os.environ['AZURE_AI_ENDPOINT']}")
    print("Agent is ready to use.")

    print("\n" + "="*50)
    print("STARTING RESEARCH TASK")
    print("="*50)

    try:
        # Run the research agent with real-time streaming
        query = "Research the latest developments in Mushroom-based Packaging"
        print(f"Query: {query}")
        print("Invoking agent with real-time output...")
        print("\n" + "="*50)
        print("REAL-TIME EXECUTION LOG")
        print("="*50)

        # Stream the agent execution to see real-time events
        result = None
        async for chunk in agent.astream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values"
        ):
            if "messages" in chunk and chunk["messages"]:
                last_message = chunk["messages"][-1]

                # Print AI messages and tool calls in real-time
                if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                    for tool_call in last_message.tool_calls:
                        tool_name = tool_call.get('name', 'unknown')
                        tool_args = tool_call.get('args', {})
                        print(f"🔧 TOOL CALL: {tool_name}")
                        if tool_name == 'task':
                            subagent_type = tool_args.get('subagent_type', 'unknown')
                            description = tool_args.get('description', '')[:100] + '...'
                            print(f"   🤖 SUBAGENT: {subagent_type}")
                            print(f"   📝 TASK: {description}")
                        elif tool_name == 'write_file':
                            file_path = tool_args.get('file_path', 'unknown')
                            content_length = len(tool_args.get('content', ''))
                            if 'critique' in file_path:
                                print(f"   📝 CRITIQUE FILE: {file_path} ({content_length} chars)")
                            else:
                                print(f"   📄 FILE: {file_path} ({content_length} chars)")
                        else:
                            print(f"   📋 ARGS: {str(tool_args)[:100]}...")

                elif hasattr(last_message, 'name') and hasattr(last_message, 'content'):
                    # Tool response
                    tool_name = last_message.name
                    content_length = len(last_message.content)
                    print(f"✅ TOOL RESPONSE: {tool_name} ({content_length} chars)")

                    # Show critique agent feedback in detail
                    if tool_name == 'task' and 'critique' in last_message.content.lower():
                        print(f"📝 CRITIQUE FEEDBACK:")
                        # Show first 300 chars of critique
                        critique_preview = last_message.content[:300] + '...' if len(last_message.content) > 300 else last_message.content
                        print(f"   {critique_preview}")

                elif hasattr(last_message, 'content') and last_message.content:
                    # AI response
                    content_preview = last_message.content[:100] + '...' if len(last_message.content) > 100 else last_message.content
                    print(f"💬 AI: {content_preview}")

            result = chunk  # Keep the last chunk as result

        print("\n" + "="*50)
        print("AGENT EXECUTION COMPLETED")
        print("="*50)

        # Check for generated files in agent state
        print("\n" + "="*50)
        print("CHECKING GENERATED FILES")
        print("="*50)

        if "files" in result and result["files"]:
            print("✅ Files found in agent state:")
            for filename, content in result["files"].items():
                print(f"  📄 {filename} ({len(content)} characters)")
                if filename.endswith('_question.txt'):
                    print(f"     Question: {content}")
                elif filename.endswith('_report.md'):
                    print(f"     Report preview: {content[:200]}...")
                elif filename.endswith('_critique.md'):
                    print(f"     📝 CRITIQUE FEEDBACK:")
                    print(f"     {content[:300]}..." if len(content) > 300 else f"     {content}")

            # Create actual files on disk
            print("\n" + "="*50)
            print("CREATING ACTUAL FILES ON DISK")
            print("="*50)

            for filename, content in result["files"].items():
                try:
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(content)
                    print(f"✅ Created actual file: {filename}")
                except Exception as e:
                    print(f"❌ Failed to create {filename}: {e}")
        else:
            print("❌ No files found in agent state")

            # Fallback: create timestamped files manually
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            print(f"Creating fallback files with timestamp: {timestamp}")

            question_file = f"{timestamp}_question.txt"
            with open(question_file, "w", encoding="utf-8") as f:
                f.write(query)
            print(f"✅ Created {question_file}")

            if result and "messages" in result:
                # Find the last substantial message content
                final_content = ""
                for msg in reversed(result["messages"]):
                    if hasattr(msg, 'content') and msg.content and len(msg.content) > 100:
                        final_content = msg.content
                        break

                if final_content:
                    report_file = f"{timestamp}_report.md"
                    with open(report_file, "w", encoding="utf-8") as f:
                        f.write(final_content)
                    print(f"✅ Created {report_file}")

        # Analyze tool calls and subagent usage
        print("\n" + "="*50)
        print("DETAILED EXECUTION ANALYSIS")
        print("="*50)

        tool_calls_count = {}
        subagent_calls = []
        errors = []

        if result and "messages" in result:
            for i, msg in enumerate(result["messages"]):
                print(f"\n--- Message {i+1}: {type(msg).__name__} ---")

                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_name = tool_call.get('name', 'unknown')
                        tool_args = tool_call.get('args', {})

                        # Count tool calls
                        tool_calls_count[tool_name] = tool_calls_count.get(tool_name, 0) + 1

                        print(f"  🔧 TOOL CALL: {tool_name}")
                        print(f"     Args: {tool_args}")

                        # Track subagent calls
                        if tool_name == 'task':
                            subagent_type = tool_args.get('subagent_type', 'unknown')
                            description = tool_args.get('description', '')[:100] + '...'
                            subagent_calls.append({
                                'type': subagent_type,
                                'description': description
                            })
                            print(f"     🤖 SUBAGENT: {subagent_type}")
                            print(f"     📝 Task: {description}")

                elif hasattr(msg, 'name') and hasattr(msg, 'content'):
                    # Tool response message
                    tool_name = msg.name
                    content = msg.content

                    print(f"  📤 TOOL RESPONSE: {tool_name}")

                    if 'Error:' in content:
                        errors.append({
                            'tool': tool_name,
                            'error': content[:200] + '...' if len(content) > 200 else content
                        })
                        print(f"     ❌ ERROR: {content[:100]}...")
                    else:
                        print(f"     ✅ SUCCESS: {len(content)} characters")
                        if tool_name == 'internet_search':
                            # Parse search results
                            try:
                                import json
                                search_data = json.loads(content)
                                if 'results' in search_data:
                                    print(f"     🔍 Found {len(search_data['results'])} search results")
                            except:
                                pass

                elif hasattr(msg, 'content') and msg.content:
                    # Regular AI message
                    content_preview = msg.content[:150] + '...' if len(msg.content) > 150 else msg.content
                    print(f"  💬 AI RESPONSE: {content_preview}")

        # Summary
        print("\n" + "="*50)
        print("EXECUTION SUMMARY")
        print("="*50)

        print(f"📊 Tool Calls Summary:")
        for tool, count in tool_calls_count.items():
            print(f"   {tool}: {count} calls")

        if subagent_calls:
            print(f"\n🤖 Subagent Calls: {len(subagent_calls)}")
            for i, call in enumerate(subagent_calls, 1):
                print(f"   {i}. {call['type']}: {call['description']}")
        else:
            print(f"\n🤖 Subagent Calls: 0 (agent used direct tools)")

        if errors:
            print(f"\n❌ Errors Encountered: {len(errors)}")
            for i, error in enumerate(errors, 1):
                print(f"   {i}. {error['tool']}: {error['error']}")
        else:
            print(f"\n✅ No errors encountered")

    except Exception as e:
        print(f"\n❌ ERROR during execution: {e}")
        import traceback
        traceback.print_exc()

    # Uncomment the following lines to run an interactive test:
    # user_query = input("\nEnter your research question (or press Enter to skip): ")
    # if user_query.strip():
    #     print(f"\nResearching: {user_query}")
    #     result = await agent.ainvoke({"messages": [{"role": "user", "content": user_query}]})
    #     print("\nResearch completed!")
    #     print("Check the generated files: question.txt and final_report.md")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
