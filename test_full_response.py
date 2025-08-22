#!/usr/bin/env python3
"""
Test script to see full RunPod response
"""

import os
import requests
import json
from dotenv import load_dotenv
import time

load_dotenv()

ENDPOINT_ID = os.getenv('RUNPOD_ENDPOINT_ID')
API_KEY = os.getenv('RUNPOD_API_KEY')

print("🔍 Testing RunPod Response")
print("=" * 50)

# Use async endpoint for better visibility
url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
headers = {"Authorization": f"Bearer {API_KEY}"}

payload = {
    "input": {
        "video": "https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4",
        "prompt": "What do you see in this video? Describe any animals or actions.",
        "fps": 1.0,
        "max_frames": 5
    }
}

print("📤 Sending request...")
response = requests.post(url, json=payload, headers=headers)

if response.status_code == 200:
    job = response.json()
    job_id = job.get('id')
    print(f"✅ Job submitted: {job_id}")
    
    # Poll for results
    status_url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{job_id}"
    
    print("\n⏳ Waiting for results...")
    for i in range(60):  # Wait up to 3 minutes
        time.sleep(3)
        
        status_response = requests.get(status_url, headers=headers)
        if status_response.status_code == 200:
            result = status_response.json()
            status = result.get('status')
            
            print(f"Status: {status}", end="\r")
            
            if status == 'COMPLETED':
                print(f"\n\n✅ COMPLETED!")
                print("\n" + "=" * 50)
                print("FULL RESPONSE:")
                print("=" * 50)
                print(json.dumps(result, indent=2))
                
                # Extract specific parts
                if 'output' in result:
                    output = result['output']
                    
                    # Handle nested output
                    if isinstance(output, dict):
                        if 'output' in output:
                            output = output['output']
                        
                        if 'caption' in output:
                            print("\n" + "=" * 50)
                            print("📝 CAPTION:")
                            print("=" * 50)
                            print(output['caption'])
                        
                        if 'error' in output:
                            print("\n❌ ERROR:", output['error'])
                    else:
                        print("\nRaw output:", output)
                
                break
                
            elif status == 'FAILED':
                print(f"\n❌ Failed!")
                print(json.dumps(result, indent=2))
                break
else:
    print(f"❌ Request failed: {response.status_code}")
    print(response.text)