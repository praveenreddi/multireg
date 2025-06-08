import asyncio
import json
import requests
import re
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_core.models import ChatCompletionClient, CreateResult, RequestUsage

class IntelligentClient(ChatCompletionClient):
    """Pure LLM intelligence - no keywords, just smart decisions"""
    
    @property
    def model_info(self):
        return {"function_calling": True}
    
    async def create(self, messages, **kwargs):
        tools = kwargs.get("tools", [])
        
        if tools:
            # Let LLM decide intelligently
            enhanced_messages = self.add_smart_instructions(messages, tools)
            result = self.call_your_api(enhanced_messages)
            
            # Check if LLM decided to use a tool
            tool_result = self.parse_and_execute_tool(result, tools)
            if tool_result:
                return tool_result
        
        # Regular response
        result = self.call_your_api(messages)
        return CreateResult(
            content=result["content"],
            usage=RequestUsage(prompt_tokens=10, completion_tokens=10)
        )
    
    def add_smart_instructions(self, messages, tools):
        """Give LLM clear instructions about available tools"""
        
        # Build tool descriptions
        tool_descriptions = []
        for tool in tools:
            if callable(tool):
                import inspect
                sig = inspect.signature(tool)
                params = list(sig.parameters.keys())
                tool_descriptions.append(
                    f"- {tool.__name__}({', '.join(params)}): {tool.__doc__ or 'Available function'}"
                )
        
        system_message = f"""You are a helpful assistant with access to these functions:

{chr(10).join(tool_descriptions)}

When a user asks something that can be answered using one of these functions, respond with:
CALL: function_name(parameter1, parameter2, ...)

For example:
- CALL: get_weather(Texas)
- CALL: calculate_sum(15, 25)
- CALL: send_email(john@example.com, Hello there)

If the question doesn't need any function, respond normally.
Use your intelligence to decide when functions are needed."""
        
        # Convert messages
        enhanced = []
        for msg in messages:
            enhanced.append({
                "role": getattr(msg, "role", "user"),
                "content": getattr(msg, "content", str(msg))
            })
        
        # Add system instruction
        enhanced.insert(0, {"role": "system", "content": system_message})
        return enhanced
    
    def parse_and_execute_tool(self, result, tools):
        """Parse LLM response and execute tool if requested"""
        
        content = result["content"]
        
        # Look for CALL: pattern
        call_match = re.search(r'CALL:\s*(\w+)\((.*?)\)', content, re.IGNORECASE)
        
        if call_match:
            function_name = call_match.group(1)
            parameters_str = call_match.group(2)
            
            print(f"🧠 LLM decided to call: {function_name}")
            print(f"🔧 Parameters: {parameters_str}")
            
            # Find the function
            target_function = None
            for tool in tools:
                if callable(tool) and tool.__name__ == function_name:
                    target_function = tool
                    break
            
            if target_function:
                try:
                    # Parse parameters
                    params = self.parse_parameters(parameters_str)
                    
                    # Execute function
                    print(f"⚡ Executing: {function_name}({params})")
                    result = target_function(*params)
                    print(f"✅ Result: {result}")
                    
                    return CreateResult(
                        content=result,
                        usage=RequestUsage(prompt_tokens=10, completion_tokens=10),
                        finish_reason="tool_calls"
                    )
                    
                except Exception as e:
                    print(f"❌ Execution error: {e}")
                    return CreateResult(
                        content=f"Function execution failed: {e}",
                        usage=RequestUsage(prompt_tokens=10, completion_tokens=10)
                    )
        
        return None
    
    def parse_parameters(self, params_str):
        """Parse function parameters from string"""
        if not params_str.strip():
            return []
        
        # Simple parameter parsing
        # Handle: "Texas", "15, 25", "john@example.com, Hello"
        params = []
        
        # Split by comma and clean up
        raw_params = [p.strip().strip('"\'') for p in params_str.split(',')]
        
        for param in raw_params:
            if param:
                params.append(param)
        
        return params
    
    def call_your_api(self, messages):
        """Your API call"""
        payload = {
            "domainName": "GenerativeAI",
            "modelName": "gpt-4-32k",
            "modelPayload": {
                "messages": messages,
                "temperature": 0,
                "max_tokens": 800
            }
        }
        
        headers = {
            'Authorization': 'Bearer your_token_here',
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.post(
                "https://your-api-url.com/question",
                headers=headers,
                data=json.dumps(payload),
                timeout=30
            )
            
            data = response.json()
            return {"content": data["modelResult"]["choices"][0]["message"]["content"]}
        except Exception as e:
            return {"content": f"API Error: {e}"}
    
    # Required methods
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

# Define your tools
def get_weather(location):
    """Get current weather information for any location"""
    weather_data = {
        "texas": "The weather in Texas is sunny and 75°F with light winds.",
        "california": "The weather in California is partly cloudy and 68°F.",
        "new york": "The weather in New York is rainy and 62°F.",
        "florida": "The weather in Florida is humid and 82°F with thunderstorms.",
        "london": "The weather in London is foggy and 55°F."
    }
    return weather_data.get(location.lower(), f"The weather in {location} is pleasant today.")

def calculate_sum(a, b):
    """Calculate the sum of two numbers"""
    try:
        result = float(a) + float(b)
        return f"The sum of {a} and {b} is {result}"
    except:
        return "Please provide valid numbers for calculation."

def send_email(recipient, message):
    """Send an email to someone (simulated)"""
    return f"Email sent to {recipient} with message: '{message}'"

def get_time():
    """Get current time"""
    from datetime import datetime
    return f"Current time is {datetime.now().strftime('%H:%M:%S')}"

def search_web(query):
    """Search the web for information"""
    return f"Here are the search results for '{query}': [Simulated search results]"

# Test the intelligent system
async def test_intelligent_tools():
    client = IntelligentClient()
    
    assistant = AssistantAgent(
        name="smart_assistant",
        system_message="You are a helpful assistant. Use available tools when appropriate.",
        model_client=client,
        tools=[get_weather, calculate_sum, send_email, get_time, search_web]
    )
    
    # Test various queries - LLM should intelligently decide
    test_queries = [
        "What's the weather like in London?",           # Should use get_weather
        "How is the climate in Texas?",                 # Should use get_weather  
        "What's 127 plus 384?",                        # Should use calculate_sum
        "Add 50 and 75",                               # Should use calculate_sum
        "Send an email to alice@company.com saying meeting at 3pm",  # Should use send_email
        "What time is it?",                            # Should use get_time
        "Search for information about Python programming",  # Should use search_web
        "Hello, how are you today?",                   # Should NOT use any tool
        "Tell me a joke",                              # Should NOT use any tool
    ]
    
    for query in test_queries:
        print(f"\n🎯 Query: '{query}'")
        print("=" * 60)
        
        team = RoundRobinGroupChat([assistant], MaxMessageTermination(2))
        result = await team.run(task=query)
        
        if result.messages:
            response = result.messages[-1].content
            print(f"📝 Response: {response}")
        
        print("-" * 60)

if __name__ == "__main__":
    asyncio.run(test_intelligent_tools())













import requests
import json

def simple_tool_test():
    """Easiest way to test if your API supports tools"""
    
    access_token = "your_access_token_here"
    llm_url = 
    
    # Your normal payload
    normal_payload = {
        "domainName": "GenerativeAI",
        "modelName": "gpt-4-32k",
        "modelPayload": {
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0,
            "max_tokens": 100
        }
    }
    
    # Same payload but WITH tools parameter
    tool_payload = {
        "domainName": "GenerativeAI",
        "modelName": "gpt-4-32k",
        "modelPayload": {
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0,
            "max_tokens": 100,
            "tools": [{"type": "function", "function": {"name": "test"}}]  # Simple tool
        }
    }
    
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    
    print("🔍 SIMPLE TOOL COMPATIBILITY TEST")
    print("=" * 40)
    
    # Test 1: Normal request (should work)
    print("1️⃣ Testing normal request...")
    try:
        response1 = requests.post(llm_url, headers=headers, data=json.dumps(normal_payload))
        print(f"   Status: {response1.status_code}")
        if response1.status_code == 200:
            print("   ✅ Normal request works")
        else:
            print("   ❌ Normal request failed")
            return
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Test 2: Request with tools parameter
    print("\n2️⃣ Testing with tools parameter...")
    try:
        response2 = requests.post(llm_url, headers=headers, data=json.dumps(tool_payload))
        print(f"   Status: {response2.status_code}")
        
        if response2.status_code == 200:
            print("   ✅ Tools parameter accepted!")
            
            # Check response content
            try:
                data = response2.json()
                response_text = str(data).lower()
                
                if "tool_call" in response_text or "function" in response_text:
                    print("   🎉 API SUPPORTS TOOLS!")
                    return True
                else:
                    print("   🤔 Tools parameter accepted but no tool calling in response")
                    return False
                    
            except:
                print("   ⚠️ Tools parameter accepted but response unclear")
                return False
                
        else:
            print("   ❌ Tools parameter rejected")
            print(f"   Response: {response2.text[:200]}...")
            return False
            
    except Exception as e:
        print(f"   ❌ Error with tools: {e}")
        return False

# Just run this one function
result = simple_tool_test()

if result:
    print("\n🎉 CONCLUSION: Your API supports tool calling!")
else:
    print("\n❌ CONCLUSION: Your API does NOT support tool calling")
    print("💡 You'll need to use OpenAI API or implement workarounds")
