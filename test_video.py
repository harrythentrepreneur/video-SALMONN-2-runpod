#!/usr/bin/env python3
"""
Test video processing on RunPod
Uses .env file for credentials
"""

import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

ENDPOINT_ID = os.getenv('RUNPOD_ENDPOINT_ID')
API_KEY = os.getenv('RUNPOD_API_KEY')

if not ENDPOINT_ID or not API_KEY:
    print("❌ Missing credentials in .env file")
    exit(1)

print("🎬 Testing video-SALMONN 2 on RunPod")
print(f"📍 Endpoint: {ENDPOINT_ID}")
print("-" * 50)

# Test with a sample video URL
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
response = requests.post(url, json=payload, headers=headers, timeout=60)

if response.status_code == 200:
    result = response.json()
    if "output" in result:
        print("\n✅ SUCCESS!")
        print("\n📝 Caption:")
        print(result['output']['caption'])
        print("\n📊 Metadata:")
        print(json.dumps(result['output'].get('metadata', {}), indent=2))
    else:
        print("\n📦 Response:")
        print(json.dumps(result, indent=2))
else:
    print(f"\n❌ Error: {response.status_code}")
    print(response.text)