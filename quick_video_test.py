#!/usr/bin/env python3
"""
Quick test to verify everything works
"""

import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

ENDPOINT_ID = os.getenv('RUNPOD_ENDPOINT_ID')
API_KEY = os.getenv('RUNPOD_API_KEY')

print("🎬 Testing video-SALMONN 2")
print("=" * 50)

url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"
headers = {"Authorization": f"Bearer {API_KEY}"}

payload = {
    "input": {
        "video": "https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4",
        "prompt": "What animals do you see in this video?",
        "fps": 2.0,
        "max_frames": 10
    }
}

print("📤 Sending request...")
response = requests.post(url, json=payload, headers=headers, timeout=120)

if response.status_code == 200:
    result = response.json()
    
    # Handle the double-wrapped output
    if "output" in result:
        output = result["output"]
        if isinstance(output, dict) and "output" in output:
            output = output["output"]
        
        if "caption" in output:
            print("\n✅ SUCCESS!")
            print("\n📝 Caption:")
            print("-" * 50)
            print(output["caption"])
            print("-" * 50)
            
            metadata = output.get("metadata", {})
            print(f"\n📊 Metadata:")
            print(f"  • Frames: {metadata.get('num_frames')}")
            print(f"  • FPS: {metadata.get('fps_used')}")
        else:
            print("Response:", json.dumps(output, indent=2))
    else:
        print("Full response:", json.dumps(result, indent=2))
else:
    print(f"❌ Error: {response.status_code}")
    print(response.text)