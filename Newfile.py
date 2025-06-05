import asyncio
import json
import requests
from typing import Any, List, Dict, Optional, Union, AsyncIterator
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.models import ChatCompletionClient, LLMMessage, CreateResult, RequestUsage

# Define predefined labels once
PREDEFINED_LABELS = [
    "unable_to_view_bill",
    "overcharged",
    "billing_history_related",
    "unable_to_download_bill",
    "billing_error",
    "virtual_agent_related",
    "chat_disappear_disconnect_issues",
    "chat_agent_related",
    "chat_not_responding",
    "customer_service_agent_related",
    "unable_to_login",
    "credentials_not_accepted",
    "unable_to_logout_find_logout_option"
]

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
    
    access_token = "your_access_token_here"
    llm_url = "https://your-api-endpoint.com/api/v1/question"
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

# Tool functions
def translate_text(text: str) -> str:
    """Translate non-English text to English"""
    return f"Translate this text to English: {text}"

def classify_text(text: str) -> str:
    """Classify customer query using predefined labels"""
    return f"Classify this text: '{text}' into one of these labels: {', '.join(PREDEFINED_LABELS)}"

async def customer_service_system(query: str):
    """Main customer service system"""
    
    client = CustomModelClient()
    
    print(f"🎯 Processing Query: {query}")
    print("=" * 50)
    
    # Create agents
    translator = AssistantAgent(
        "translator",
        system_message="You are a translator. Translate non-English text to English. If already in English, return as-is. Be brief and accurate.",
        model_client=client,
        tools=[translate_text]
    )
    
    classifier = AssistantAgent(
        "classifier",
        system_message="You are a text classifier. Use the classify_text tool and return ONLY the exact label name from the predefined list.",
        model_client=client,
        tools=[classify_text]
    )
    
    # Create team
    team = RoundRobinGroupChat([translator, classifier])
    
    try:
        print("🚀 Starting translation and classification...")
        result = await team.run(task=f"Process this customer query: {query}")
        print("\n✅ Processing Complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    await client.close()

async def main():
    """Test the system"""
    test_queries = [
        "Este proyecto de ley es demasiado caro",
        "I cannot see my bill",
        "Chat agent disconnected",
        "Cannot login to my account"
    ]
    
    await customer_service_system(test_queries[0])

if __name__ == "__main__":
    asyncio.run(main())
