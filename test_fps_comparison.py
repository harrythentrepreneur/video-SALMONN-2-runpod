#!/usr/bin/env python3
"""
Compare different FPS settings
"""

import os
import requests
import json
import time
from dotenv import load_dotenv

load_dotenv()

ENDPOINT_ID = os.getenv('RUNPOD_ENDPOINT_ID')
API_KEY = os.getenv('RUNPOD_API_KEY')

# Test video URL
VIDEO_URL = "https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4"

print("🎬 Testing Different FPS Settings")
print("=" * 50)

# Test different FPS values
fps_tests = [0.5, 1.0, 2.0, 5.0]

for fps in fps_tests:
    print(f"\n📊 Testing FPS: {fps}")
    print("-" * 30)
    
    url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    payload = {
        "input": {
            "video": VIDEO_URL,
            "prompt": "What happens in this video?",
            "fps": fps,
            "max_frames": 20
        }
    }
    
    start = time.time()
    response = requests.post(url, json=payload, headers=headers, timeout=120)
    elapsed = time.time() - start
    
    if response.status_code == 200:
        result = response.json()
        if "output" in result:
            metadata = result['output'].get('metadata', {})
            caption = result['output']['caption'][:100] + "..."
            
            print(f"✅ Success!")
            print(f"  • Frames: {metadata.get('num_frames', 'N/A')}")
            print(f"  • Time: {elapsed:.1f}s")
            print(f"  • Caption: {caption}")
        else:
            print(f"⚠️ Unexpected response")
    else:
        print(f"❌ Error: {response.status_code}")
    
    time.sleep(2)  # Pause between tests

print("\n" + "=" * 50)
print("✅ FPS Testing Complete!")