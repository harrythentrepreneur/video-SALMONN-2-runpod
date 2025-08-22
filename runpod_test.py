#!/usr/bin/env python3
"""
Minimal RunPod handler for testing
This will verify RunPod can connect and run
"""

import runpod
import time
import json

def handler(event):
    """Simple test handler"""
    print("🎯 RunPod handler called!")
    print(f"Received event: {json.dumps(event, indent=2)}")
    
    # Get input
    input_data = event.get('input', {})
    
    # Simulate some processing
    time.sleep(2)
    
    # Return test response
    return {
        "output": {
            "status": "success",
            "message": "RunPod connection working! Model handler is ready.",
            "received_input": input_data,
            "timestamp": time.time()
        }
    }

if __name__ == "__main__":
    print("Starting RunPod test handler...")
    runpod.serverless.start({"handler": handler})