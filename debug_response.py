#!/usr/bin/env python3
"""
Debug what RunPod is actually returning
"""

import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

ENDPOINT_ID = os.getenv('RUNPOD_ENDPOINT_ID')
API_KEY = os.getenv('RUNPOD_API_KEY')

print("🔍 Debugging RunPod Response Format")
print("=" * 50)

url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"
headers = {"Authorization": f"Bearer {API_KEY}"}

payload = {
    "input": {
        "video": "https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4",
        "prompt": "Test",
        "fps": 1.0,
        "max_frames": 3
    }
}

print("Sending minimal test request...")
response = requests.post(url, json=payload, headers=headers, timeout=180)

print(f"\nStatus Code: {response.status_code}")
print("\nRaw Response Text (first 2000 chars):")
print("-" * 50)
print(response.text[:2000])

try:
    result = response.json()
    print("\n\nParsed JSON Structure:")
    print("-" * 50)
    
    def show_structure(obj, indent=0):
        """Show JSON structure with types"""
        spaces = "  " * indent
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    print(f"{spaces}{key}: {type(value).__name__}")
                    show_structure(value, indent + 1)
                else:
                    val_str = str(value)[:100] if value else "None"
                    print(f"{spaces}{key}: {type(value).__name__} = {val_str}")
        elif isinstance(obj, list):
            print(f"{spaces}[list with {len(obj)} items]")
            if obj:
                show_structure(obj[0], indent + 1)
    
    show_structure(result)
    
    # Save full response for inspection
    with open('debug_response.json', 'w') as f:
        json.dump(result, f, indent=2)
    print("\n✅ Full response saved to debug_response.json")
    
except Exception as e:
    print(f"\n❌ Failed to parse JSON: {e}")