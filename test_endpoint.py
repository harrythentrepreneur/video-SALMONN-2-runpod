#!/usr/bin/env python3
"""
Test RunPod endpoint with environment variables
"""

import os
import requests
import json
import time
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get configuration from .env
ENDPOINT_ID = os.getenv('RUNPOD_ENDPOINT_ID')
API_KEY = os.getenv('RUNPOD_API_KEY')

if not ENDPOINT_ID or not API_KEY:
    print("❌ Missing configuration!")
    print("Please create a .env file with:")
    print("  RUNPOD_ENDPOINT_ID=your_endpoint_id")
    print("  RUNPOD_API_KEY=your_api_key")
    exit(1)

def test_with_sample_video():
    """Test with a sample video URL"""
    print("🎬 Testing video-SALMONN 2 on RunPod")
    print(f"📍 Endpoint: {ENDPOINT_ID}")
    print("=" * 50)
    
    # Use async endpoint for better handling
    url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
    
    payload = {
        "input": {
            "video": "https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4",
            "prompt": "What animals and actions do you see in this video? Describe in detail.",
            "fps": 2.0,  # 2 frames per second
            "max_frames": 20,
            "temperature": 0.7
        }
    }
    
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    print("📤 Submitting job...")
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code != 200:
        print(f"❌ Failed to submit: {response.text}")
        return
    
    job = response.json()
    job_id = job.get('id')
    print(f"✅ Job ID: {job_id}")
    
    # Poll for results
    status_url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{job_id}"
    
    print("\n⏳ Processing... (this may take 1-2 minutes)")
    start_time = time.time()
    
    while True:
        time.sleep(3)
        status_response = requests.get(status_url, headers=headers)
        
        if status_response.status_code != 200:
            print(f"❌ Status check failed: {status_response.text}")
            break
        
        result = status_response.json()
        status = result.get('status')
        
        elapsed = int(time.time() - start_time)
        print(f"\r⏳ Status: {status} ({elapsed}s)", end='', flush=True)
        
        if status == 'COMPLETED':
            print(f"\n✅ Completed in {elapsed} seconds!\n")
            
            output = result.get('output', {})
            
            if 'caption' in output:
                print("📝 VIDEO CAPTION:")
                print("=" * 50)
                print(output['caption'])
                print("=" * 50)
                
                metadata = output.get('metadata', {})
                print(f"\n📊 Processing Stats:")
                print(f"  • Frames analyzed: {metadata.get('num_frames', 'N/A')}")
                print(f"  • FPS used: {metadata.get('fps_used', 'N/A')}")
                print(f"  • Processing time: {metadata.get('processing_time', 'N/A')}s")
            elif 'error' in output:
                print(f"❌ Error: {output['error']}")
            else:
                print("📦 Raw output:")
                print(json.dumps(output, indent=2))
            break
            
        elif status == 'FAILED':
            print(f"\n❌ Job failed!")
            print(json.dumps(result, indent=2))
            break
            
        elif elapsed > 300:  # 5 minute timeout
            print(f"\n⏱️ Timeout after 5 minutes")
            break

def test_with_local_video(video_path):
    """Test with a local video file"""
    print(f"📹 Loading local video: {video_path}")
    
    with open(video_path, "rb") as f:
        video_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    print(f"📦 Video size: {len(video_base64) / 1024 / 1024:.2f} MB (base64)")
    
    url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
    
    payload = {
        "input": {
            "video": video_base64,
            "prompt": "Describe everything you see in this video in detail.",
            "fps": 3.0,  # Higher FPS for local video
            "max_frames": 30,
            "temperature": 0.7
        }
    }
    
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    print("📤 Uploading and processing...")
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code != 200:
        print(f"❌ Failed: {response.text}")
        return
    
    job_id = response.json().get('id')
    print(f"✅ Job ID: {job_id}")
    print("Check status at: https://www.runpod.io/console/serverless")

if __name__ == "__main__":
    import sys
    
    # Install python-dotenv if needed
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("📦 Installing python-dotenv...")
        os.system("pip install python-dotenv")
        from dotenv import load_dotenv
    
    if len(sys.argv) > 1:
        # Test with local video
        test_with_local_video(sys.argv[1])
    else:
        # Test with sample video
        test_with_sample_video()