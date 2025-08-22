#!/usr/bin/env python3
"""
Test RunPod with your own video
"""

import os
import sys
import requests
import json
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

ENDPOINT_ID = os.getenv('RUNPOD_ENDPOINT_ID')
API_KEY = os.getenv('RUNPOD_API_KEY')

def test_with_file(video_path, fps=2.0):
    """Test with a local video file"""
    
    print(f"📹 Testing with: {video_path}")
    print(f"⚙️ FPS setting: {fps}")
    print("-" * 50)
    
    # Read and encode video
    with open(video_path, "rb") as f:
        video_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    print(f"📦 Video size: {len(video_base64) / 1024 / 1024:.2f} MB (encoded)")
    
    # Send request
    url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    payload = {
        "input": {
            "video": video_base64,
            "prompt": "Describe what happens in this video in detail",
            "fps": fps,
            "max_frames": 30
        }
    }
    
    print("📤 Sending to RunPod...")
    response = requests.post(url, json=payload, headers=headers, timeout=120)
    
    if response.status_code == 200:
        result = response.json()
        if "output" in result:
            print("\n✅ SUCCESS!")
            print("\n📝 Video Caption:")
            print("-" * 50)
            print(result['output']['caption'])
            print("-" * 50)
            
            metadata = result['output'].get('metadata', {})
            print(f"\n📊 Processing Stats:")
            print(f"  • Frames analyzed: {metadata.get('num_frames', 'N/A')}")
            print(f"  • FPS used: {metadata.get('fps_used', 'N/A')}")
        else:
            print("\n📦 Response:")
            print(json.dumps(result, indent=2))
    else:
        print(f"\n❌ Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_my_video.py <video_file> [fps]")
        print("Example: python test_my_video.py video.mp4 2.0")
        sys.exit(1)
    
    video_file = sys.argv[1]
    fps = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    
    if not os.path.exists(video_file):
        print(f"❌ File not found: {video_file}")
        sys.exit(1)
    
    test_with_file(video_file, fps)