#!/usr/bin/env python3
"""
Debug test to see what RunPod is receiving
"""

import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

ENDPOINT_ID = os.getenv('RUNPOD_ENDPOINT_ID')
API_KEY = os.getenv('RUNPOD_API_KEY')

print("🔍 Debug Test for RunPod")
print("=" * 50)

# Test with minimal payload
url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"
headers = {"Authorization": f"Bearer {API_KEY}"}

# Try different payload formats
payloads = [
    {
        "name": "Standard format",
        "data": {
            "input": {
                "video": "https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4",
                "prompt": "Test",
                "fps": 1.0,
                "max_frames": 5
            }
        }
    },
    {
        "name": "Direct format",
        "data": {
            "video": "https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4",
            "prompt": "Test",
            "fps": 1.0,
            "max_frames": 5
        }
    }
]

for test in payloads:
    print(f"\nTesting: {test['name']}")
    print(f"Payload: {json.dumps(test['data'], indent=2)}")
    print("-" * 30)
    
    response = requests.post(url, json=test['data'], headers=headers, timeout=60)
    
    print(f"Status: {response.status_code}")
    
    try:
        result = response.json()
        if "output" in result:
            print("✅ Has output field")
            if "debug" in result["output"]:
                print(f"Debug info: {result['output']['debug']}")
        elif "caption" in result:
            print("✅ Has caption field")
        elif "error" in result:
            print(f"❌ Error: {result['error']}")
        
        print(f"Response keys: {list(result.keys())}")
        
        # Show first part of response
        response_str = json.dumps(result, indent=2)
        print(f"Response: {response_str[:500]}...")
    except:
        print(f"Raw response: {response.text[:500]}")

print("\n" + "=" * 50)
print("Check RunPod logs to see what the handler received")