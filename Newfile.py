import asyncio
import json
import requests
from typing import Any, List, Dict, Optional, Union, AsyncIterator
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
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
    """Complete Custom Model Client with ALL abstract methods"""
    
    def __init__(self):
        self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._actual_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """Model information and capabilities"""
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
    def capabilities(self) -> Dict[str, Any]:
        """Model capabilities"""
        return {
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "structured_output": False,
        }

    async def create(
        self, 
        messages: List[LLMMessage], 
        **kwargs: Any
    ) -> CreateResult:
        """Create completion"""
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
    
    async def create_stream(
        self, 
        messages: List[LLMMessage], 
        **kwargs: Any
    ) -> AsyncIterator[str]:
        """Create streaming completion"""
        result = await self.create(messages, **kwargs)
        yield result.content
    
    @property
    def actual_usage(self) -> RequestUsage:
        """Get actual usage from last request"""
        return self._actual_usage
    
    @property
    def total_usage(self) -> RequestUsage:
        """Get total usage across all requests"""
        return self._total_usage
    
    @property
    def remaining_tokens(self) -> int:
        """Get remaining tokens in context"""
        return max(0, 8000 - self._total_usage.prompt_tokens - self._total_usage.completion_tokens)
    
    def count_tokens(
        self, 
        messages: List[LLMMessage], 
        **kwargs: Any
    ) -> int:
        """Count tokens in messages"""
        total = 0
        for msg in messages:
            if hasattr(msg, 'content'):
                content = msg.content
            else:
                content = str(msg)
            # Rough token estimation
            total += len(content.split()) * 1.3
        return int(total)
    
    async def close(self) -> None:
        """Close the client"""
        pass

# Tool functions with strict label enforcement
def translate_text(text: str) -> str:
    """Translate non-English text to English"""
    return f"Please translate this text to English: '{text}'"

def classify_text(text: str) -> str:
    """Classify customer query using ONLY predefined labels"""
    labels_str = "\n".join([f"- {label}" for label in PREDEFINED_LABELS])
    return f"""Classify this text: '{text}'

You MUST choose EXACTLY ONE label from this list:
{labels_str}

Return ONLY the label name, nothing else. Do not create new labels."""

async def customer_service_roundrobin(query: str):
    """RoundRobin with proper termination and ALL abstracts"""
    
    client = CustomModelClient()
    
    print(f"🎯 Processing Query: {query}")
    print("=" * 50)
    
    # Create translator agent
    translator = AssistantAgent(
        "translator",
        system_message="""You are a translator. 
        - If text is in English, say "Already in English: [text]"
        - If text is not in English, translate it to English
        - Be brief and accurate
        - End your response with "TRANSLATION_DONE" """,
        model_client=client,
        tools=[translate_text]
    )
    
    # Create classifier agent with STRICT label enforcement
    classifier = AssistantAgent(
        "classifier",
        system_message=f"""You are a text classifier. You MUST classify text using ONLY these predefined labels:

{chr(10).join([f"- {label}" for label in PREDEFINED_LABELS])}

RULES:
- Choose EXACTLY ONE label from the list above
- Return ONLY the label name
- Do NOT create new labels
- Do NOT use variations of the labels
- End your response with "CLASSIFICATION_DONE" """,
        model_client=client,
        tools=[classify_text]
    )
    
    # RoundRobin with termination after 4 messages (2 per agent)
    team = RoundRobinGroupChat(
        [translator, classifier],
        termination_condition=MaxMessageTermination(max_messages=4)
    )
    
    try:
        print("🚀 Starting RoundRobin processing...")
        print(f"📊 Initial Token Usage: {client.total_usage}")
        
        result = await team.run(task=f"""Process this customer service query: "{query}"

Step 1 - Translator: Translate if needed
Step 2 - Classifier: Classify using predefined labels only""")
        
        print(f"📊 Final Token Usage: {client.total_usage}")
        print(f"📊 Remaining Tokens: {client.remaining_tokens}")
        print("\n✅ RoundRobin Complete!")
        print("=" * 50)
        
        # Extract results from messages
        messages = result.messages
        translation = "N/A"
        classification = "N/A"
        
        for msg in messages:
            content = msg.content.lower()
            if "translation_done" in content:
                translation = msg.content.replace("TRANSLATION_DONE", "").strip()
            elif "classification_done" in content:
                classification = msg.content.replace("CLASSIFICATION_DONE", "").strip()
        
        print("🎉 FINAL RESULTS:")
        print(f"📝 Original: {query}")
        print(f"🌐 Translation: {translation}")
        print(f"🏷️ Classification: {classification}")
        print("=" * 50)
        
        return {
            "original": query,
            "translated": translation,
            "classification": classification,
            "token_usage": client.total_usage
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    
    finally:
        await client.close()

async def batch_process_queries(queries: List[str]):
    """Process multiple queries"""
    results = []
    
    print("🚀 BATCH PROCESSING STARTED")
    print("=" * 60)
    
    for i, query in enumerate(queries, 1):
        print(f"\n📋 Query {i}/{len(queries)}")
        result = await customer_service_roundrobin(query)
        if result:
            results.append(result)
        
        if i < len(queries):
            print("\n" + "="*60)
    
    print("\n🎉 BATCH PROCESSING COMPLETE!")
    print("=" * 60)
    
    return results

async def main():
    """Test the system with ALL abstracts"""
    test_queries = [
        "Este proyecto de ley es demasiado caro",  # Should be "overcharged"
        "I cannot see my bill",                    # Should be "unable_to_view_bill"
        "Chat agent disconnected",                 # Should be "chat_disappear_disconnect_issues"
        "Cannot login to my account"               # Should be "unable_to_login"
    ]
    
    print("🎯 Customer Service System - AutoGen v0.4 with ALL Abstracts")
    print("=" * 70)
    
    # Process single query
    await customer_service_roundrobin(test_queries[0])
    
    # Uncomment to process all queries
    # results = await batch_process_queries(test_queries)
    # print(f"\n📊 Processed {len(results)} queries successfully!")

if __name__ == "__main__":
    asyncio.run(main())
