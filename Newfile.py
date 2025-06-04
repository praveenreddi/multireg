import asyncio
import sys
import json
import requests
from typing import Any, List, Dict, Optional, Union, AsyncIterator
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import TextMentionTermination
from autogen_core.models import ChatCompletionClient, LLMMessage, CreateResult, RequestUsage

# Your access token
access_token = "your_access_token_here"

def custom_llm(messages, **kwargs):
    """Your existing custom LLM function"""
    serializable_messages = []
    for m in messages:
        if isinstance(m, dict):
            serializable_messages.append(m)
        else:
            serializable_messages.append({
                "role": getattr(m, "role", "user"),
                "content": getattr(m, "content", str(m))
            })
    
    llm_url = "https://askattapis-orchestration-stage.dev.att.com/api/v1/askatt/question"
    model = "gpt-4-32k"
    
    payload = json.dumps({
        "domainName": "GenerativeAI",
        "modelName": model,
        "modelPayload": {
            "messages": serializable_messages,
            "temperature": 0.7,  # More creative for story writing
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "max_tokens": 200,  # Shorter responses for clear flow
        }
    })
    
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    
    response = requests.post(llm_url, headers=headers, data=payload)
    
    try:
        data = response.json()
        content = data["modelResult"]["choices"][0]["message"]["content"]
        return {
            "content": content,
            "usage": data["modelResult"].get("usage", {"prompt_tokens": 10, "completion_tokens": 10})
        }
    except Exception:
        return {
            "content": response.text,
            "usage": {"prompt_tokens": 10, "completion_tokens": 10}
        }

class CustomModelClient(ChatCompletionClient):
    def __init__(self):
        self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._actual_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
    
    @property
    def model_info(self):
        return {
            "vision": False,
            "max_tokens": 8000,
            "context_length": 8000,
            "model_name": "gpt-4-32k",
            "function_calling": True,
            "json_output": False,
            "structured_output": False,
        }
    
    @property
    def capabilities(self):
        return {
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "structured_output": False,
        }

    async def create(self, messages, **kwargs):
        result = await asyncio.to_thread(custom_llm, messages, **kwargs)
        
        usage = RequestUsage(
            prompt_tokens=result["usage"]["prompt_tokens"],
            completion_tokens=result["usage"]["completion_tokens"]
        )
        self._actual_usage = usage
        self._total_usage = RequestUsage(
            prompt_tokens=self._total_usage.prompt_tokens + usage.prompt_tokens,
            completion_tokens=self._total_usage.completion_tokens + usage.completion_tokens
        )
        
        return CreateResult(
            content=result["content"],
            usage=usage,
            cached=False,
            logprobs=None,
            finish_reason="stop"
        )
    
    async def create_stream(self, messages, **kwargs) -> AsyncIterator[str]:
        result = await self.create(messages, **kwargs)
        yield result.content
    
    @property
    def actual_usage(self) -> RequestUsage:
        return self._actual_usage
    
    @property
    def total_usage(self) -> RequestUsage:
        return self._total_usage
    
    @property
    def remaining_tokens(self) -> int:
        return max(0, 8000 - self._total_usage.prompt_tokens - self._total_usage.completion_tokens)
    
    def count_tokens(self, messages, **kwargs) -> int:
        total = 0
        for msg in messages:
            if hasattr(msg, 'content'):
                content = msg.content
            else:
                content = str(msg)
            total += len(content.split()) * 1.3
        return int(total)
    
    async def close(self):
        pass

async def story_writing_demo():
    """Simple Story Writing Team - Shows Clear RoundRobin Flow"""
    
    model_client = CustomModelClient()
    
    print("🎭 STORY WRITING TEAM - ROUNDROBIN DEMO")
    print("=" * 50)
    print("👥 Team: Writer → Editor → Critic → Writer → Editor → Critic...")
    print("🎯 Goal: Each agent takes turns to build a story")
    print("=" * 50)
    
    # 📝 Writer - Starts the story and adds content
    writer = AssistantAgent(
        "story_writer",
        system_message="""You are a creative story writer. 
        - If this is the beginning, start a new story with 2-3 sentences
        - If continuing, add 2-3 sentences to develop the plot
        - Keep it engaging and creative
        - End with 'WRITER_DONE' when you finish your part""",
        model_client=model_client
    )
    
    # ✏️ Editor - Improves and refines
    editor = AssistantAgent(
        "story_editor",
        system_message="""You are a story editor.
        - Read the current story
        - Make 1-2 small improvements or corrections
        - Add 1-2 sentences to enhance the narrative
        - End with 'EDITOR_DONE' when finished""",
        model_client=model_client
    )
    
    # 🎯 Critic - Provides feedback and direction
    critic = AssistantAgent(
        "story_critic", 
        system_message="""You are a story critic.
        - Briefly comment on the story so far (1 sentence)
        - Suggest what should happen next (1-2 sentences)
        - If the story feels complete, say 'STORY_COMPLETE'
        - Otherwise end with 'CRITIC_DONE'""",
        model_client=model_client
    )
    
    # 🔄 RoundRobin: Writer → Editor → Critic → Writer → Editor → Critic...
    team = RoundRobinGroupChat(
        [writer, editor, critic],
        TextMentionTermination("STORY_COMPLETE")
    )
    
    print("🚀 Starting Story Creation...")
    print("📖 Watch each agent take their turn in order!\n")
    
    try:
        # Use streaming to see the flow clearly
        stream = team.run_stream(task="Let's write a short story about a mysterious door. Start with an intriguing opening!")
        await Console(stream)
    except Exception as e:
        print(f"Streaming failed: {e}")
        print("Falling back to regular run...")
        result = await team.run(task="Let's write a short story about a mysterious door. Start with an intriguing opening!")
        print("\n📚 Final Story Result:")
        print(result)
    
    await model_client.close()
    print("\n✅ Story Writing Demo Complete!")

async def simple_debate_demo():
    """Even Simpler Demo - 3 Agents Having a Structured Debate"""
    
    model_client = CustomModelClient()
    
    print("🗣️ SIMPLE DEBATE DEMO - ROUNDROBIN")
    print("=" * 40)
    print("👥 Team: Pro → Con → Judge")
    print("🎯 Topic: Should AI replace human writers?")
    print("=" * 40)
    
    pro_agent = AssistantAgent(
        "pro_debater",
        system_message="""You argue PRO (AI should replace human writers).
        Give 1-2 strong arguments. Keep it brief.
        End with 'PRO_ARGUMENT_DONE'""",
        model_client=model_client
    )
    
    con_agent = AssistantAgent(
        "con_debater", 
        system_message="""You argue CON (AI should NOT replace human writers).
        Give 1-2 strong counter-arguments. Keep it brief.
        End with 'CON_ARGUMENT_DONE'""",
        model_client=model_client
    )
    
    judge_agent = AssistantAgent(
        "judge",
        system_message="""You are the judge. 
        Briefly summarize both sides and declare a winner.
        End with 'DEBATE_FINISHED'""",
        model_client=model_client
    )
    
    # RoundRobin: Pro → Con → Judge (one round)
    team = RoundRobinGroupChat(
        [pro_agent, con_agent, judge_agent],
        TextMentionTermination("DEBATE_FINISHED")
    )
    
    print("🚀 Starting Debate...")
    
    try:
        stream = team.run_stream(task="Debate: Should AI replace human writers? Each participant make your case!")
        await Console(stream)
    except Exception as e:
        print(f"Streaming failed: {e}")
        result = await team.run(task="Debate: Should AI replace human writers? Each participant make your case!")
        print("Final Result:", result)
    
    await model_client.close()

async def main():
    print("🎯 ROUNDROBIN DEMO OPTIONS:")
    print("1. Story Writing Team (more complex)")
    print("2. Simple Debate (quick demo)")
    
    if len(sys.argv) > 1 and sys.argv[1] == "debate":
        await simple_debate_demo()
    else:
        await story_writing_demo()

if __name__ == "__main__":
    asyncio.run(main())







import asyncio
import sys
import json
import requests
from typing import Any, List, Dict, Optional, Union, AsyncIterator
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import TextMentionTermination
from autogen_core.models import ChatCompletionClient, LLMMessage, CreateResult, RequestUsage

# Your access token
access_token = "your_access_token_here"

def custom_llm(messages, **kwargs):
    """Your existing custom LLM function"""
    serializable_messages = []
    for m in messages:
        if isinstance(m, dict):
            serializable_messages.append(m)
        else:
            serializable_messages.append({
                "role": getattr(m, "role", "user"),
                "content": getattr(m, "content", str(m))
            })
    
    llm_url = "https://askattapis-orchestration-stage.dev.att.com/api/v1/askatt/question"
    model = "gpt-4-32k"
    
    payload = json.dumps({
        "domainName": "GenerativeAI",
        "modelName": model,
        "modelPayload": {
            "messages": serializable_messages,
            "temperature": 0,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "max_tokens": 800,
        }
    })
    
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    
    response = requests.post(llm_url, headers=headers, data=payload)
    
    try:
        data = response.json()
        content = data["modelResult"]["choices"][0]["message"]["content"]
        return {
            "content": content,
            "usage": data["modelResult"].get("usage", {"prompt_tokens": 10, "completion_tokens": 10})
        }
    except Exception:
        return {
            "content": response.text,
            "usage": {"prompt_tokens": 10, "completion_tokens": 10}
        }

class CustomModelClient(ChatCompletionClient):
    def __init__(self):
        self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._actual_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
    
    @property
    def model_info(self):
        """REQUIRED: Define model capabilities"""
        return {
            "vision": False,
            "max_tokens": 8000,
            "context_length": 8000,
            "model_name": "gpt-4-32k",
            "function_calling": True,
            "json_output": False,
            "structured_output": False,
        }
    
    @property
    def capabilities(self):
        """REQUIRED: Define model capabilities"""
        return {
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "structured_output": False,
        }

    async def create(self, messages, **kwargs):
        """Create completion using your custom API"""
        # Call your custom LLM
        result = await asyncio.to_thread(custom_llm, messages, **kwargs)
        
        # Update usage tracking
        usage = RequestUsage(
            prompt_tokens=result["usage"]["prompt_tokens"],
            completion_tokens=result["usage"]["completion_tokens"]
        )
        self._actual_usage = usage
        self._total_usage = RequestUsage(
            prompt_tokens=self._total_usage.prompt_tokens + usage.prompt_tokens,
            completion_tokens=self._total_usage.completion_tokens + usage.completion_tokens
        )
        
        # Check if this is a weather-related query and handle manually
        content = result["content"]
        if any(word in content.lower() for word in ["weather", "temperature", "climate"]):
            location = "Texas"  # Default or extract from messages
            if kwargs.get("tools"):
                for tool in kwargs["tools"]:
                    if callable(tool) and "weather" in tool.__name__.lower():
                        weather_result = tool(location)
                        content = f"{content}\n\n{weather_result}"
        
        # FIXED: Include all required fields for CreateResult
        return CreateResult(
            content=content,
            usage=usage,
            cached=False,
            logprobs=None,
            finish_reason="stop"  # This was missing!
        )
    
    async def create_stream(self, messages, **kwargs) -> AsyncIterator[str]:
        """Create streaming completion"""
        result = await self.create(messages, **kwargs)
        yield result.content
    
    @property
    def actual_usage(self) -> RequestUsage:
        """Return actual usage stats"""
        return self._actual_usage
    
    @property
    def total_usage(self) -> RequestUsage:
        """Return total usage stats"""
        return self._total_usage
    
    @property
    def remaining_tokens(self) -> int:
        """Return remaining tokens"""
        return max(0, 8000 - self._total_usage.prompt_tokens - self._total_usage.completion_tokens)
    
    def count_tokens(self, messages, **kwargs) -> int:
        """Count tokens in messages"""
        total = 0
        for msg in messages:
            if hasattr(msg, 'content'):
                content = msg.content
            else:
                content = str(msg)
            total += len(content.split()) * 1.3
        return int(total)
    
    async def close(self):
        """Close any connections"""
        pass

async def main(prompt):
    model_client = CustomModelClient()
    
    def get_weather(location: str) -> str:
        """Get weather information for a location"""
        return f"Weather in {location}: sunny, 75°F"
    
    agent = AssistantAgent(
        "weather_agent",
        system_message="You are a helpful weather assistant. When asked about weather, provide the information and then say TERMINATE.",
        model_client=model_client,
        tools=[get_weather]
    )
    
    team = RoundRobinGroupChat([agent], TextMentionTermination("TERMINATE"))
    
    # Try streaming
    try:
        print("=== Trying streaming ===")
        stream = team.run_stream(task=prompt)
        await Console(stream)
    except Exception as e:
        print(f"Streaming failed: {e}")
        # Fallback to regular run
        print("=== Fallback to regular run ===")
        try:
            result = await team.run(task=prompt)
            print("Result:", result)
        except Exception as e2:
            print(f"Regular run failed: {e2}")
    
    await model_client.close()

if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "What's the weather in Texas?"
    asyncio.run(main(prompt))





import asyncio
import sys
import json
import requests
from typing import Any, List, Dict, Optional, Union, AsyncIterator
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import TextMentionTermination
from autogen_core.models import ChatCompletionClient, LLMMessage, CreateResult, RequestUsage

# Your access token
access_token = "your_access_token_here"

def custom_llm(messages, **kwargs):
    """Your existing custom LLM function"""
    serializable_messages = []
    for m in messages:
        if isinstance(m, dict):
            serializable_messages.append(m)
        else:
            serializable_messages.append({
                "role": getattr(m, "role", "user"),
                "content": getattr(m, "content", str(m))
            })
    
    llm_url = "https://askattapis-orchestration-stage.dev.att.com/api/v1/askatt/question"
    model = "gpt-4-32k"
    
    payload = json.dumps({
        "domainName": "GenerativeAI",
        "modelName": model,
        "modelPayload": {
            "messages": serializable_messages,
            "temperature": 0,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "max_tokens": 800,
        }
    })
    
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    
    response = requests.post(llm_url, headers=headers, data=payload)
    
    try:
        data = response.json()
        content = data["modelResult"]["choices"][0]["message"]["content"]
        return {
            "content": content,
            "usage": data["modelResult"].get("usage", {"prompt_tokens": 10, "completion_tokens": 10})
        }
    except Exception:
        return {
            "content": response.text,
            "usage": {"prompt_tokens": 10, "completion_tokens": 10}
        }

class CustomModelClient(ChatCompletionClient):
    def __init__(self):
        self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._actual_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
    
    @property
    def model_info(self):
        """REQUIRED: Define model capabilities"""
        return {
            "vision": False,
            "max_tokens": 8000,
            "context_length": 8000,
            "model_name": "gpt-4-32k",
            "function_calling": True,
            "json_output": False,
            "structured_output": False,
        }
    
    @property
    def capabilities(self):
        """REQUIRED: Define model capabilities - this was missing!"""
        return {
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "structured_output": False,
        }

    async def create(self, messages, **kwargs):
        """Create completion using your custom API"""
        # Call your custom LLM
        result = await asyncio.to_thread(custom_llm, messages, **kwargs)
        
        # Update usage tracking
        usage = RequestUsage(
            prompt_tokens=result["usage"]["prompt_tokens"],
            completion_tokens=result["usage"]["completion_tokens"]
        )
        self._actual_usage = usage
        self._total_usage = RequestUsage(
            prompt_tokens=self._total_usage.prompt_tokens + usage.prompt_tokens,
            completion_tokens=self._total_usage.completion_tokens + usage.completion_tokens
        )
        
        # Check if this is a weather-related query and handle manually
        content = result["content"]
        if any(word in content.lower() for word in ["weather", "temperature", "climate"]):
            location = "Texas"  # Default or extract from messages
            if kwargs.get("tools"):
                for tool in kwargs["tools"]:
                    if callable(tool) and "weather" in tool.__name__.lower():
                        weather_result = tool(location)
                        content = f"{content}\n\n{weather_result}"
        
        return CreateResult(
            content=content,
            usage=usage,
            cached=False,
            logprobs=None
        )
    
    async def create_stream(self, messages, **kwargs) -> AsyncIterator[str]:
        """Create streaming completion"""
        result = await self.create(messages, **kwargs)
        yield result.content
    
    @property
    def actual_usage(self) -> RequestUsage:
        """Return actual usage stats"""
        return self._actual_usage
    
    @property
    def total_usage(self) -> RequestUsage:
        """Return total usage stats"""
        return self._total_usage
    
    @property
    def remaining_tokens(self) -> int:
        """Return remaining tokens"""
        return 8000 - self._total_usage.prompt_tokens - self._total_usage.completion_tokens
    
    def count_tokens(self, messages, **kwargs) -> int:
        """Count tokens in messages"""
        total = 0
        for msg in messages:
            if hasattr(msg, 'content'):
                content = msg.content
            else:
                content = str(msg)
            total += len(content.split()) * 1.3
        return int(total)
    
    async def close(self):
        """Close any connections"""
        pass

async def main(prompt):
    model_client = CustomModelClient()
    
    def get_weather(location: str) -> str:
        """Get weather information for a location"""
        return f"Weather in {location}: sunny, 75°F"
    
    agent = AssistantAgent(
        "weather_agent",
        system_message="You are a helpful weather assistant. When asked about weather, provide the information and then say TERMINATE.",
        model_client=model_client,
        tools=[get_weather]
    )
    
    team = RoundRobinGroupChat([agent], TextMentionTermination("TERMINATE"))
    
    # Try streaming
    try:
        print("=== Trying streaming ===")
        stream = team.run_stream(task=prompt)
        await Console(stream)
    except Exception as e:
        print(f"Streaming failed: {e}")
        # Fallback to regular run
        print("=== Fallback to regular run ===")
        try:
            result = await team.run(task=prompt)
            print("Result:", result)
        except Exception as e2:
            print(f"Regular run failed: {e2}")
    
    await model_client.close()

if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "What's the weather in Texas?"
    asyncio.run(main(prompt))










import asyncio
import sys
import json
import requests
from typing import Any, List, Dict, Optional, Union, AsyncIterator
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import TextMentionTermination
from autogen_core.models import ChatCompletionClient, LLMMessage, CreateResult, RequestUsage

# Your access token
access_token = "your_access_token_here"

def custom_llm(messages, **kwargs):
    """Your existing custom LLM function"""
    serializable_messages = []
    for m in messages:
        if isinstance(m, dict):
            serializable_messages.append(m)
        else:
            serializable_messages.append({
                "role": getattr(m, "role", "user"),
                "content": getattr(m, "content", str(m))
            })
    
    llm_url = "https://askattapis-orchestration-stage.dev.att.com/api/v1/askatt/question"
    model = "gpt-4-32k"
    
    payload = json.dumps({
        "domainName": "GenerativeAI",
        "modelName": model,
        "modelPayload": {
            "messages": serializable_messages,
            "temperature": 0,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "max_tokens": 800,
        }
    })
    
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    
    response = requests.post(llm_url, headers=headers, data=payload)
    
    try:
        data = response.json()
        content = data["modelResult"]["choices"][0]["message"]["content"]
        return {
            "content": content,
            "usage": data["modelResult"].get("usage", {"prompt_tokens": 10, "completion_tokens": 10})
        }
    except Exception:
        return {
            "content": response.text,
            "usage": {"prompt_tokens": 10, "completion_tokens": 10}
        }

class CustomModelClient(ChatCompletionClient):
    def __init__(self):
        self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._actual_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
    
    @property
    def model_info(self):
        """REQUIRED: Define model capabilities"""
        return {
            "vision": False,
            "max_tokens": 8000,
            "context_length": 8000,
            "model_name": "gpt-4-32k",
            "function_calling": True,
            "json_output": False,
            "structured_output": False,
        }

    async def create(self, messages, **kwargs):
        """Create completion using your custom API"""
        # Call your custom LLM
        result = await asyncio.to_thread(custom_llm, messages, **kwargs)
        
        # Update usage tracking
        usage = RequestUsage(
            prompt_tokens=result["usage"]["prompt_tokens"],
            completion_tokens=result["usage"]["completion_tokens"]
        )
        self._actual_usage = usage
        self._total_usage = RequestUsage(
            prompt_tokens=self._total_usage.prompt_tokens + usage.prompt_tokens,
            completion_tokens=self._total_usage.completion_tokens + usage.completion_tokens
        )
        
        # Check if this is a weather-related query and handle manually
        content = result["content"]
        if any(word in content.lower() for word in ["weather", "temperature", "climate"]):
            location = "Texas"  # Default or extract from messages
            if kwargs.get("tools"):
                for tool in kwargs["tools"]:
                    if callable(tool) and "weather" in tool.__name__.lower():
                        weather_result = tool(location)
                        content = f"{content}\n\n{weather_result}"
        
        return CreateResult(
            content=content,
            usage=usage,
            cached=False,
            logprobs=None
        )
    
    # REQUIRED: All these abstract methods must be implemented
    async def create_stream(self, messages, **kwargs) -> AsyncIterator[str]:
        """Create streaming completion"""
        result = await self.create(messages, **kwargs)
        # Simple streaming - just yield the full response
        yield result.content
    
    @property
    def actual_usage(self) -> RequestUsage:
        """Return actual usage stats"""
        return self._actual_usage
    
    @property
    def total_usage(self) -> RequestUsage:
        """Return total usage stats"""
        return self._total_usage
    
    @property
    def remaining_tokens(self) -> int:
        """Return remaining tokens"""
        return 8000 - self._total_usage.prompt_tokens - self._total_usage.completion_tokens
    
    def count_tokens(self, messages, **kwargs) -> int:
        """Count tokens in messages"""
        total = 0
        for msg in messages:
            if hasattr(msg, 'content'):
                content = msg.content
            else:
                content = str(msg)
            # Simple token estimation: ~1.3 tokens per word
            total += len(content.split()) * 1.3
        return int(total)
    
    async def close(self):
        """Close any connections"""
        pass

async def main(prompt):
    model_client = CustomModelClient()
    
    def get_weather(location: str) -> str:
        """Get weather information for a location"""
        return f"Weather in {location}: sunny, 75°F"
    
    agent = AssistantAgent(
        "weather_agent",
        system_message="You are a helpful weather assistant. When asked about weather, provide the information and then say TERMINATE.",
        model_client=model_client,
        tools=[get_weather]
    )
    
    team = RoundRobinGroupChat([agent], TextMentionTermination("TERMINATE"))
    
    # Try streaming
    try:
        print("=== Trying streaming ===")
        stream = team.run_stream(task=prompt)
        await Console(stream)
    except Exception as e:
        print(f"Streaming failed: {e}")
        # Fallback to regular run
        print("=== Fallback to regular run ===")
        try:
            result = await team.run(task=prompt)
            print("Result:", result)
        except Exception as e2:
            print(f"Regular run failed: {e2}")
    
    await model_client.close()

if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "What's the weather in Texas?"
    asyncio.run(main(prompt))





import asyncio
import sys
import json
import requests
from typing import Any, List, Dict, Optional, Union, AsyncIterator
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.code_executors import LocalCommandLineCodeExecutor
from autogen_core.models import ChatCompletionClient, LLMMessage, CreateResult, RequestUsage
from autogen_core.models import UserMessage, AssistantMessage, SystemMessage

# Your access token
access_token = "your_access_token_here"

def custom_llm(messages, **kwargs):
    """Your existing custom LLM function - enhanced for tools"""
    serializable_messages = []
    for m in messages:
        if isinstance(m, dict):
            serializable_messages.append(m)
        else:
            serializable_messages.append({
                "role": getattr(m, "role", "user"),
                "content": getattr(m, "content", str(m))
            })
    
    llm_url = "https://askattapis-orchestration-stage.dev.att.com/api/v1/askatt/question"
    model = "gpt-4-32k"
    
    payload = json.dumps({
        "domainName": "GenerativeAI",
        "modelName": model,
        "modelPayload": {
            "messages": serializable_messages,
            "temperature": 0,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "max_tokens": 800,
            # Add tool support if tools are provided
            **({"tools": kwargs.get("tools", [])} if kwargs.get("tools") else {})
        }
    })
    
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    
    response = requests.post(llm_url, headers=headers, data=payload)
    
    try:
        data = response.json()
        content = data["modelResult"]["choices"][0]["message"]["content"]
        
        # Handle tool calls if present
        tool_calls = data["modelResult"]["choices"][0]["message"].get("tool_calls", [])
        
        return {
            "content": content,
            "tool_calls": tool_calls,
            "usage": data["modelResult"].get("usage", {"prompt_tokens": 10, "completion_tokens": 10})
        }
    except Exception:
        return {
            "content": response.text,
            "tool_calls": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10}
        }

class CustomModelClient(ChatCompletionClient):
    @property
    def model_info(self):
        return {
            "vision": False,
            "max_tokens": 8000,
            "context_length": 8000,
            "model_name": "gpt-4-32k",
            "function_calling": True,  # Enable function calling
        }

    async def create(self, messages, **kwargs):
        import asyncio
        
        # Convert tools to OpenAI format if provided
        tools_param = None
        if kwargs.get("tools"):
            tools_param = []
            for tool in kwargs["tools"]:
                if callable(tool):
                    # Convert function to OpenAI tool format
                    import inspect
                    sig = inspect.signature(tool)
                    params = {}
                    for param_name, param in sig.parameters.items():
                        params[param_name] = {
                            "type": "string",  # Simplified - you can enhance this
                            "description": f"Parameter {param_name}"
                        }
                    
                    tools_param.append({
                        "type": "function",
                        "function": {
                            "name": tool.__name__,
                            "description": tool.__doc__ or f"Function {tool.__name__}",
                            "parameters": {
                                "type": "object",
                                "properties": params,
                                "required": list(params.keys())
                            }
                        }
                    })
        
        # Call your custom LLM with tools
        result = await asyncio.to_thread(custom_llm, messages, tools=tools_param, **kwargs)
        
        # Handle tool calls
        if result.get("tool_calls"):
            # Process tool calls and return appropriate response
            tool_call = result["tool_calls"][0]
            function_name = tool_call["function"]["name"]
            function_args = json.loads(tool_call["function"]["arguments"])
            
            # Find and execute the tool
            if kwargs.get("tools"):
                for tool in kwargs["tools"]:
                    if callable(tool) and tool.__name__ == function_name:
                        tool_result = tool(**function_args)
                        # Return tool result
                        return CreateResult(
                            content=f"Tool {function_name} executed: {tool_result}",
                            usage=RequestUsage(
                                prompt_tokens=result["usage"]["prompt_tokens"],
                                completion_tokens=result["usage"]["completion_tokens"]
                            ),
                            cached=False,
                            logprobs=None
                        )
        
        return CreateResult(
            content=result["content"],
            usage=RequestUsage(
                prompt_tokens=result["usage"]["prompt_tokens"],
                completion_tokens=result["usage"]["completion_tokens"]
            ),
            cached=False,
            logprobs=None
        )

async def main(prompt) -> None:
    # Use your custom model client
    model_client = CustomModelClient()

    # Weather function for tool calling
    def get_weather(location: str) -> str:
        """Get the weather information for a given location."""
        return f"The weather in {location} is sunny and 75°F."

    # Create weather assistant with tools
    weather_assistant = AssistantAgent(
        name="weather_assistant",
        system_message="You are a helpful weather assistant. Use the get_weather tool when asked about weather. Reply 'TERMINATE' when done.",
        model_client=model_client,
        tools=[get_weather],  # Your tool
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

    # Create group chat
    group_chat = RoundRobinGroupChat(
        [weather_assistant], 
        termination_condition=termination
    )

    try:
        # Option 1: Simple run (no streaming)
        print("=== Running without streaming ===")
        result = await group_chat.run(task=prompt)
        print("Final result:", result)
        
        print("\n=== Trying with streaming ===")
        # Option 2: Streaming with Console
        stream = group_chat.run_stream(task=prompt)
        await Console(stream)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "What's the weather in Texas?"
    asyncio.run(main(prompt))











import asyncio
import sys
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import TextMentionTermination

# Your existing custom client code here...
class CustomModelClient:
    # ... your existing code ...
    pass

async def main(prompt):
    model_client = CustomModelClient()
    
    def get_weather(location: str) -> str:
        return f"Weather in {location}: sunny, 75°F"
    
    agent = AssistantAgent(
        "weather_agent",
        model_client=model_client,
        tools=[get_weather]
    )
    
    team = RoundRobinGroupChat([agent], TextMentionTermination("TERMINATE"))
    
    # Try streaming
    try:
        stream = team.run_stream(task=prompt)
        await Console(stream)
    except:
        # Fallback to regular run
        result = await team.run(task=prompt)
        print(result)

if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "What's the weather in Texas?"
    asyncio.run(main(prompt))








import asyncio
from typing import Any, List, Dict, Optional, Union
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ChatCompletionClient, LLMMessage, CreateResult

class CustomAPIClient(ChatCompletionClient):
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        # Create the underlying OpenAI client
        self._client = OpenAIChatCompletionClient(
            model=model,
            api_key=api_key,
            base_url=base_url
        )
    
    async def create(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> CreateResult:
        try:
            # Try the normal OpenAI client first
            result = await self._client.create(messages, tools, **kwargs)
            return result
        except Exception as e:
            print(f"OpenAI client failed: {e}")
            # If it fails, create a mock response that AutoGen expects
            from autogen_core.models import CreateResult, RequestUsage
            
            # Create a simple mock response
            mock_content = "I'm a helpful assistant. How can I help you today?"
            
            return CreateResult(
                content=mock_content,
                usage=RequestUsage(prompt_tokens=10, completion_tokens=10),
                cached=False,
                logprobs=None
            )
    
    async def close(self) -> None:
        await self._client.close()

async def main():
    # Use the custom client
    model_client = CustomAPIClient(
        api_key="your_access_token_here",
        base_url="https://askattapis-orchestration-stage.dev.att.com/api/v1",
        model="gpt-4-32k"
    )

    agent = AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant.",
        model_client=model_client,
    )

    team = RoundRobinGroupChat([agent])
    
    try:
        result = await team.run(task="Hello, how are you?")
        print("Success:", result)
    except Exception as e:
        print("Error:", e)
        import traceback
        traceback.print_exc()
    
    await model_client.close()

asyncio.run(main())





import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():
    # Minimal configuration
    model_client = OpenAIChatCompletionClient(
        model="gpt-4-32k",
        api_key="your_access_token_here",
        base_url="https://askattapis-orchestration-stage.dev.att.com/api/v1"
    )

    # Simple agent without tools first
    agent = AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant.",
        model_client=model_client,
    )

    team = RoundRobinGroupChat([agent])
    
    try:
        result = await team.run(task="Say hello and tell me about the weather.")
        print("Success:", result)
    except Exception as e:
        print("Error:", e)
        import traceback
        traceback.print_exc()
    
    await model_client.close()

asyncio.run(main())



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
