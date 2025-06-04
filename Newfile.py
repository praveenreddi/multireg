# Replace the try block with this for streaming:
try:
    stream = group_chat.run_stream(task=prompt)
    await Console(stream)
except Exception as e:
    print(f"Streaming failed: {e}")
    # Fallback to regular run
    result = await group_chat.run(task=prompt)
    print("Result:", result)






import asyncio
import sys
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.code_executors import LocalCommandLineCodeExecutor

# Your working access token
access_token = "your_access_token_here"

async def main(prompt) -> None:
    # Use the SAME configuration that worked in your test
    model_client = OpenAIChatCompletionClient(
        model="gpt-4-32k",
        api_key=access_token,
        base_url="https://askattapis-orchestration-stage.dev.att.com/api/v1"
    )

    # Weather function - SIMPLE format that works
    def get_weather(location: str) -> str:
        """Get the weather information for a given location."""
        return f"The weather in {location} is sunny and 75°F."

    # Create weather assistant - FIXED tool format
    weather_assistant = AssistantAgent(
        name="weather_assistant",
        system_message="You are a helpful weather assistant. Use the get_weather tool when asked about weather. Reply 'TERMINATE' when done.",
        model_client=model_client,
        tools=[get_weather],  # Direct function - NOT FunctionTool
    )

    # Create assistant agent
    assistant = AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant. Write all code in python. Reply 'TERMINATE' if the task is done.",
        model_client=model_client,
    )

    # Create code executor agent
    code_executor = CodeExecutorAgent(
        name="code_executor",
        code_executor=LocalCommandLineCodeExecutor(work_dir="coding"),
    )

    # Set up termination
    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(10)

    # Create group chat with weather assistant
    group_chat = RoundRobinGroupChat(
        [weather_assistant], 
        termination_condition=termination
    )

    try:
        # FIXED streaming - simple approach
        result = await group_chat.run(task=prompt)
        print("Final result:", result)
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        await model_client.close()

if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "What's the weather in Texas?"
    asyncio.run(main(prompt))



import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():
    model_client = OpenAIChatCompletionClient(
        model="gpt-4-32k",
        api_key="your_token",
        base_url="https://askattapis-orchestration-stage.dev.att.com/api/v1/",
    )

    def get_weather(location: str) -> str:
        return f"Weather in {location}: sunny, 75°F"

    agent = AssistantAgent(
        "weather_agent",
        model_client=model_client,
        tools=[get_weather]
    )

    team = RoundRobinGroupChat([agent], TextMentionTermination("TERMINATE"))
    await Console(team.run_stream(task="What's the weather in Texas?"))
    await model_client.close()

asyncio.run(main())







import asyncio
import sys
import logging
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.code_executors import LocalCommandLineCodeExecutor

# Configure your access token
access_token = "your_access_token_here"

async def main(prompt) -> None:
    # Define a model client
    model_client = OpenAIChatCompletionClient(
        model="gpt-4-32k",
        api_key=access_token,
        base_url="https://askattapis-orchestration-stage.dev.att.com/api/v1/",
        max_retries=7,
        model_info={
            "vision": True,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
            "structured_output": True,
        },
    )

    # Create assistant agent
    assistant = AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant. Write all code in python. Reply only 'TERMINATE' if the task is done.",
        model_client=model_client,
    )

    # Create code executor agent
    code_executor = CodeExecutorAgent(
        name="code_executor",
        code_executor=LocalCommandLineCodeExecutor(work_dir="coding"),
    )

    # Define weather function - FIXED FORMAT
    def get_weather(location: str) -> str:
        """Get the weather information for a given location and return it."""
        return f"temperature in {location} is sunny."

    # Create weather assistant with CORRECT tool format
    weather_assistant = AssistantAgent(
        name="weather_assistant",
        system_message="You are a helpful assistant. If the user asks about temperature or weather, always use the get_weather tool.",
        model_client=model_client,
        tools=[get_weather],  # FIXED: Direct function reference
    )

    # Set up termination condition
    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(10)

    # Create group chat
    group_chat = RoundRobinGroupChat(
        [weather_assistant], 
        termination_condition=termination
    )

    # Run with Console for proper streaming
    stream = group_chat.run_stream(task=prompt)
    await Console(stream)
    
    # Close model client
    await model_client.close()

if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Hows the weather in texas state."
    asyncio.run(main(prompt))
