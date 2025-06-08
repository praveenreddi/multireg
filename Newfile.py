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
