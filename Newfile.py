import asyncio
import json
import requests
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_core.models import ChatCompletionClient, CreateResult, RequestUsage

class SimpleToolClient(ChatCompletionClient):
    """Simple client focused ONLY on tool testing"""
    
    @property
    def model_info(self):
        return {"function_calling": True, "max_tokens": 8000}
    
    @property
    def capabilities(self):
        return {"function_calling": True}
    
    async def create(self, messages, **kwargs):
        """Simple create - just return a basic response"""
        try:
            # For tool testing, return a simple response
            return CreateResult(
                content="I'll help you with that.",
                usage=RequestUsage(prompt_tokens=10, completion_tokens=10)
            )
        except Exception as e:
            return CreateResult(
                content=f"Error: {e}",
                usage=RequestUsage(prompt_tokens=10, completion_tokens=10)
            )
    
    # Required methods (minimal)
    async def create_stream(self, messages, **kwargs):
        result = await self.create(messages, **kwargs)
        yield result.content
    
    @property
    def actual_usage(self):
        return RequestUsage(prompt_tokens=0, completion_tokens=0)
    
    @property
    def total_usage(self):
        return RequestUsage(prompt_tokens=0, completion_tokens=0)
    
    @property
    def remaining_tokens(self):
        return 8000
    
    def count_tokens(self, messages, **kwargs):
        return 100
    
    async def close(self):
        pass

# Your tools for testing
def get_weather(location: str) -> str:
    """Get weather for a location"""
    print(f"🔧 TOOL CALLED: get_weather(location='{location}')")
    return f"The weather in {location} is sunny and 75°F."

def calculate_sum(a: str, b: str) -> str:
    """Add two numbers"""
    print(f"🔧 TOOL CALLED: calculate_sum(a='{a}', b='{b}')")
    try:
        result = float(a) + float(b)
        return f"The sum of {a} and {b} is {result}"
    except:
        return "Invalid numbers"

def send_message(recipient: str, message: str) -> str:
    """Send a message"""
    print(f"🔧 TOOL CALLED: send_message(recipient='{recipient}', message='{message}')")
    return f"Message sent to {recipient}: '{message}'"

# Test ONLY tool functionality
async def test_tools_only():
    """Test if tools are being registered and called"""
    
    print("🎯 TESTING TOOL FUNCTIONALITY ONLY")
    print("=" * 50)
    
    client = SimpleToolClient()
    
    # Create assistant WITH tools
    assistant = AssistantAgent(
        name="tool_tester",
        system_message="You are a tool testing assistant.",
        model_client=client,
        tools=[get_weather, calculate_sum, send_message]  # ← TOOLS HERE
    )
    
    print(f"✅ Assistant created with {len(assistant.tools)} tools")
    
    # List the tools
    if hasattr(assistant, 'tools') and assistant.tools:
        print("📋 Registered tools:")
        for i, tool in enumerate(assistant.tools, 1):
            if hasattr(tool, '__name__'):
                print(f"   {i}. {tool.__name__}")
            else:
                print(f"   {i}. {tool}")
    else:
        print("❌ No tools found!")
    
    # Try to manually call a tool (direct test)
    print("\n🔧 MANUAL TOOL TEST:")
    try:
        result = get_weather("Texas")
        print(f"✅ Manual call successful: {result}")
    except Exception as e:
        print(f"❌ Manual call failed: {e}")
    
    # Test with AutoGen team
    print("\n🤖 AUTOGEN TEAM TEST:")
    try:
        team = RoundRobinGroupChat([assistant], MaxMessageTermination(2))
        result = await team.run(task="Test message")
        
        if result.messages:
            print(f"✅ Team response: {result.messages[-1].content}")
        else:
            print("❌ No team response")
            
    except Exception as e:
        print(f"❌ Team test failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 TOOL TEST COMPLETE")

if __name__ == "__main__":
    asyncio.run(test_tools_only())
