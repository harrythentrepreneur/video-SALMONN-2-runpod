#!/usr/bin/env python3
"""
Quick test script for your RunPod endpoint
"""

import requests
import base64
import json
import sys

# === FILL IN YOUR DETAILS HERE ===
ENDPOINT_ID = "YOUR_ENDPOINT_ID"  # e.g., "abcdef123456"
API_KEY = "YOUR_API_KEY"  # Your RunPod API key

# RunPod endpoint URL
URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"

def test_with_video(video_path):
    """Test with a real video file"""
    
    print(f"📹 Loading video: {video_path}")
    
    # Read and encode video
    with open(video_path, "rb") as f:
        video_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    print(f"📦 Video size: {len(video_base64) / 1024 / 1024:.2f} MB (base64)")
    
    # Create request
    payload = {
        "input": {
            "video": video_base64,
            "prompt": "What is happening in this video? Describe what you see.",
            "fps": 2.0,  # Analyze 2 frames per second
            "max_frames": 30,
            "temperature": 0.7
        }
    }
    
    # Send request
    print("🚀 Sending to RunPod...")
    
    try:
        response = requests.post(
            URL,
            json=payload,
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=300  # 5 minute timeout
        )
        
        result = response.json()
        
        if response.status_code == 200:
            if "output" in result:
                print("\n✅ SUCCESS!")
                print("\n📝 CAPTION:")
                print("-" * 50)
                print(result['output']['caption'])
                print("-" * 50)
                
                metadata = result['output'].get('metadata', {})
                print(f"\n📊 Stats:")
                print(f"  - Frames processed: {metadata.get('num_frames', 'N/A')}")
                print(f"  - FPS used: {metadata.get('fps_used', 'N/A')}")
                print(f"  - Processing time: {metadata.get('processing_time', 'N/A')}s")
            else:
                print("\n⚠️ Unexpected response format:")
                print(json.dumps(result, indent=2))
        else:
            print(f"\n❌ Error (Status {response.status_code}):")
            print(json.dumps(result, indent=2))
            
    except Exception as e:
        print(f"\n❌ Request failed: {e}")

def test_with_url():
    """Test with a sample video URL"""
    
    print("🌐 Testing with sample video URL...")
    
    # You can also pass a URL if your handler supports it
    payload = {
        "input": {
            "video": "https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4",
            "prompt": "What animals do you see in this video?",
            "fps": 1.0
        }
    }
    
    print("🚀 Sending to RunPod...")
    
    try:
        response = requests.post(
            URL,
            json=payload,
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=300
        )
        
        result = response.json()
        print("\n📝 Response:")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"\n❌ Request failed: {e}")

if __name__ == "__main__":
    print("🎬 RunPod video-SALMONN 2 Test")
    print("=" * 50)
    
    if ENDPOINT_ID == "YOUR_ENDPOINT_ID":
        print("❌ Please edit this file and add your ENDPOINT_ID and API_KEY first!")
        print("   You can find these in your RunPod console")
        sys.exit(1)
    
    # Test with a local video file
    video_file = "test_video.mp4"  # Change this to your video file
    
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
    
    import os
    if os.path.exists(video_file):
        test_with_video(video_file)
    else:
        print(f"⚠️ Video file '{video_file}' not found")
        print("Usage: python test_my_endpoint.py [video_file.mp4]")
        print("\nTrying with sample URL instead...")
        test_with_url()