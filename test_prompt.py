#!/usr/bin/env python3
"""
Test script for Bedrock Prompt endpoint
"""
import requests
import json

# API endpoint
BASE_URL = "http://localhost:8000"

def test_prompt_endpoint():
    """Test the /chat/prompt endpoint with streaming"""
    
    # Test with the first prompt ARN
    prompt_arn = "arn:aws:bedrock:ap-northeast-2:125814533785:prompt/X61RA20825"
    
    # Or test with just the ID
    # prompt_id = "2CDUVN332V"
    
    payload = {
        "prompt_id": prompt_arn,
        "user_query": "너의 명언 알려줘",
        "variables": {}  # Optional: add any additional variables here
    }
    
    print(f"Testing prompt endpoint: {BASE_URL}/chat/prompt")
    print(f"Payload: {json.dumps(payload, ensure_ascii=False, indent=2)}\n")
    
    try:
        response = requests.post(
            f"{BASE_URL}/chat/prompt",
            json=payload,
            stream=True,
            timeout=60
        )
        
        print(f"Status Code: {response.status_code}\n")
        
        if response.status_code == 200:
            print("Streaming response:")
            print("-" * 80)
            
            full_text = ""
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data = json.loads(line_str[6:])
                        
                        if data['type'] == 'content':
                            text = data['text']
                            print(text, end='', flush=True)
                            full_text += text
                        
                        elif data['type'] == 'done':
                            print("\n" + "-" * 80)
                            print(f"Stream complete! Total length: {len(full_text)} characters")
                        
                        elif data['type'] == 'error':
                            print(f"\nError: {data['message']}")
            
            print("\n")
        else:
            print(f"Error: {response.text}")
    
    except Exception as e:
        print(f"Request failed: {str(e)}")

def test_health_endpoint():
    """Test the health endpoint"""
    print(f"Testing health endpoint: {BASE_URL}/health\n")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    except Exception as e:
        print(f"Request failed: {str(e)}\n")

if __name__ == "__main__":
    print("=" * 80)
    print("AWS Bedrock Prompt Endpoint Test")
    print("=" * 80)
    print()
    
    # First check if server is healthy
    test_health_endpoint()
    
    # Then test the prompt endpoint
    test_prompt_endpoint()
    
    print("\nTest complete!")
