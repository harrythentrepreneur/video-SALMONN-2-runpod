#!/usr/bin/env python3
"""
Minimal RunPod handler - just to verify deployment works
"""

import runpod
import json
import os

print("=" * 50)
print("RunPod Minimal Handler Starting...")
print(f"Python path: {os.sys.executable}")
print(f"Working directory: {os.getcwd()}")
print("=" * 50)

def handler(event):
    """Minimal handler that just echoes back"""
    print(f"Received event: {json.dumps(event, indent=2)}")
    
    input_data = event.get('input', {})
    
    # Simple response
    response = {
        "output": {
            "status": "✅ RunPod is working!",
            "message": "Your endpoint is successfully deployed and responding",
            "received": input_data,
            "info": {
                "handler": "runpod_minimal.py",
                "cuda_available": str(os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')),
                "model_path": os.environ.get('MODEL_PATH', 'not set')
            }
        }
    }
    
    print(f"Sending response: {json.dumps(response, indent=2)}")
    return response

if __name__ == "__main__":
    print("Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})