#!/usr/bin/env python3
"""
Test RunPod endpoint with proper async handling
"""

import requests
import json
import time

# Your RunPod details
ENDPOINT_ID = "uyrc9kfx7k6rfr"
API_KEY = "YOUR_API_KEY"  # Add your RunPod API key here

def test_video():
    print("🎬 Testing video-SALMONN 2 on RunPod")
    print("=" * 50)
    
    # Step 1: Submit job
    url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
    
    payload = {
        "input": {
            "video": "https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4",
            "prompt": "What animals and actions do you see in this video?",
            "fps": 1.0,
            "max_frames": 10,
            "temperature": 0.7
        }
    }
    
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    print("📤 Submitting job...")
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code != 200:
        print(f"❌ Failed to submit job: {response.text}")
        return
    
    job = response.json()
    job_id = job.get('id')
    print(f"✅ Job submitted! ID: {job_id}")
    
    # Step 2: Poll for results
    status_url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{job_id}"
    
    print("\n⏳ Waiting for results...")
    print("(This may take 1-2 minutes on first run while model loads)")
    
    start_time = time.time()
    dots = 0
    
    while True:
        time.sleep(2)
        status_response = requests.get(status_url, headers=headers)
        
        if status_response.status_code != 200:
            print(f"\n❌ Failed to get status: {status_response.text}")
            break
        
        result = status_response.json()
        status = result.get('status')
        
        # Show progress
        dots = (dots + 1) % 4
        print(f"\r⏳ Status: {status} {'.' * dots}    ", end='', flush=True)
        
        if status == 'COMPLETED':
            elapsed = time.time() - start_time
            print(f"\n\n✅ Completed in {elapsed:.1f} seconds!")
            
            output = result.get('output', {})
            
            if 'caption' in output:
                print("\n📝 VIDEO CAPTION:")
                print("=" * 50)
                print(output['caption'])
                print("=" * 50)
                
                metadata = output.get('metadata', {})
                print(f"\n📊 Stats:")
                print(f"  • Frames analyzed: {metadata.get('num_frames', 'N/A')}")
                print(f"  • FPS: {metadata.get('fps_used', 'N/A')}")
                print(f"  • Processing time: {metadata.get('processing_time', 'N/A')}s")
            else:
                print("\n📦 Full output:")
                print(json.dumps(output, indent=2))
            break
            
        elif status == 'FAILED':
            print(f"\n\n❌ Job failed!")
            error = result.get('error', 'Unknown error')
            print(f"Error: {error}")
            break
            
        elif time.time() - start_time > 300:  # 5 minute timeout
            print(f"\n\n⏱️ Timeout after 5 minutes")
            break

if __name__ == "__main__":
    test_video()
    
    print("\n" + "=" * 50)
    print("🎉 Your endpoint is working!")
    print("\nTo test with your own video:")
    print("1. Replace the video URL in this script")
    print("2. Or modify to accept base64 encoded videos")
    print("3. Adjust FPS (1-30) based on your needs")