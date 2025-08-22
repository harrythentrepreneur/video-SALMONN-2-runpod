#!/usr/bin/env python3
"""
Test script for video-SALMONN 2 RunPod deployment
"""

import os
import sys
import json
import base64
import requests
import argparse
import time
from typing import Dict, Any

def test_local_handler():
    """Test the handler locally without RunPod"""
    print("Testing local handler...")
    
    # Import the handler
    from runpod_serverless import handler
    
    # Create test event
    test_event = {
        "input": {
            "prompt": "Describe this video in detail",
            "fps": 1.0,
            "max_frames": 10,
            "temperature": 0.7,
            "process_audio": False
        }
    }
    
    # Add test video (you'll need to provide a path)
    test_video_path = "test_videos/sample.mp4"
    if os.path.exists(test_video_path):
        test_event["input"]["video"] = test_video_path
    else:
        print(f"Warning: Test video not found at {test_video_path}")
        print("Creating mock video data...")
        test_event["input"]["video"] = "mock_video_data"
    
    # Run handler
    try:
        result = handler(test_event)
        print("\nHandler Response:")
        print(json.dumps(result, indent=2))
        return result
    except Exception as e:
        print(f"Error running handler: {e}")
        return None

def test_runpod_endpoint(endpoint_url: str, api_key: str, video_path: str):
    """Test deployed RunPod endpoint"""
    print(f"Testing RunPod endpoint: {endpoint_url}")
    
    # Read and encode video
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return None
    
    with open(video_path, "rb") as f:
        video_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    print(f"Video size: {len(video_base64) / 1024 / 1024:.2f} MB (base64)")
    
    # Prepare request payload
    payload = {
        "input": {
            "video": video_base64,
            "prompt": "What is happening in this video? Describe the scene, actions, and any important details.",
            "fps": 2.0,
            "max_frames": 30,
            "temperature": 0.7,
            "max_tokens": 512,
            "process_audio": True
        }
    }
    
    # Headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Send request
    print("\nSending request to RunPod...")
    start_time = time.time()
    
    try:
        response = requests.post(
            endpoint_url,
            json=payload,
            headers=headers,
            timeout=300  # 5 minute timeout
        )
        
        elapsed_time = time.time() - start_time
        print(f"Request completed in {elapsed_time:.2f} seconds")
        
        # Parse response
        result = response.json()
        
        if response.status_code == 200:
            print("\n✅ Success!")
            if "output" in result:
                print(f"\nCaption: {result['output']['caption']}")
                print(f"\nMetadata:")
                metadata = result['output'].get('metadata', {})
                for key, value in metadata.items():
                    print(f"  - {key}: {value}")
            else:
                print("Warning: No output in response")
                print(json.dumps(result, indent=2))
        else:
            print(f"\n❌ Error (Status {response.status_code}):")
            print(json.dumps(result, indent=2))
        
        return result
        
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None

def test_fps_variations(endpoint_url: str, api_key: str, video_path: str):
    """Test different FPS settings"""
    print("Testing FPS variations...")
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    # Read video once
    with open(video_path, "rb") as f:
        video_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    # Test different FPS values
    fps_values = [0.5, 1.0, 2.0, 5.0, 10.0]
    results = []
    
    for fps in fps_values:
        print(f"\n--- Testing FPS: {fps} ---")
        
        payload = {
            "input": {
                "video": video_base64,
                "prompt": "Briefly describe this video",
                "fps": fps,
                "max_frames": 30,
                "temperature": 0.7,
                "max_tokens": 256
            }
        }
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                endpoint_url,
                json=payload,
                headers=headers,
                timeout=300
            )
            elapsed_time = time.time() - start_time
            
            result = response.json()
            
            if "output" in result:
                metadata = result['output'].get('metadata', {})
                results.append({
                    "fps_requested": fps,
                    "fps_used": metadata.get('fps_used'),
                    "num_frames": metadata.get('num_frames'),
                    "processing_time": elapsed_time,
                    "caption_length": len(result['output'].get('caption', ''))
                })
                print(f"✅ Success - Frames: {metadata.get('num_frames')}, Time: {elapsed_time:.2f}s")
            else:
                print(f"❌ Failed")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Summary
    print("\n=== FPS Test Summary ===")
    print(f"{'FPS':<10} {'Frames':<10} {'Time (s)':<12} {'Caption Len':<12}")
    print("-" * 44)
    for r in results:
        print(f"{r['fps_requested']:<10.1f} {r['num_frames']:<10} {r['processing_time']:<12.2f} {r['caption_length']:<12}")

def main():
    parser = argparse.ArgumentParser(description='Test video-SALMONN 2 RunPod deployment')
    parser.add_argument('--mode', choices=['local', 'runpod', 'fps'], default='local',
                      help='Test mode: local handler, RunPod endpoint, or FPS variations')
    parser.add_argument('--endpoint', type=str, help='RunPod endpoint URL')
    parser.add_argument('--api-key', type=str, help='RunPod API key')
    parser.add_argument('--video', type=str, default='test_videos/sample.mp4',
                      help='Path to test video file')
    
    args = parser.parse_args()
    
    if args.mode == 'local':
        test_local_handler()
    
    elif args.mode == 'runpod':
        if not args.endpoint or not args.api_key:
            print("Error: --endpoint and --api-key required for RunPod testing")
            sys.exit(1)
        test_runpod_endpoint(args.endpoint, args.api_key, args.video)
    
    elif args.mode == 'fps':
        if not args.endpoint or not args.api_key:
            print("Error: --endpoint and --api-key required for FPS testing")
            sys.exit(1)
        test_fps_variations(args.endpoint, args.api_key, args.video)

if __name__ == "__main__":
    main()