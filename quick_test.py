#!/usr/bin/env python3
"""
Quick test for your RunPod endpoint
"""

import requests
import json

# Your RunPod details
ENDPOINT_ID = "uyrc9kfx7k6rfr"
API_KEY = "YOUR_API_KEY"  # Add your RunPod API key here

# RunPod endpoint URL
URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"

def test_endpoint():
    """Test with a sample video URL"""
    
    print("🎬 Testing your video-SALMONN 2 endpoint...")
    print(f"📍 Endpoint: {ENDPOINT_ID}")
    print("-" * 50)
    
    # Using a small sample video for testing
    payload = {
        "input": {
            "video": "https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4",
            "prompt": "What is happening in this video? Describe what you see.",
            "fps": 1.0,  # 1 frame per second for quick test
            "max_frames": 10,  # Limit frames for faster processing
            "temperature": 0.7
        }
    }
    
    if API_KEY == "YOUR_API_KEY":
        print("\n⚠️  IMPORTANT: You need to add your API key!")
        print("1. Go to: https://www.runpod.io/console/user/settings")
        print("2. Click 'API Keys' → 'Create API Key'")
        print("3. Copy the key and replace YOUR_API_KEY in this file")
        print("\nTrying without auth (will likely fail)...")
        headers = {}
    else:
        headers = {"Authorization": f"Bearer {API_KEY}"}
    
    print("\n🚀 Sending request...")
    
    try:
        response = requests.post(
            URL,
            json=payload,
            headers=headers,
            timeout=300
        )
        
        result = response.json()
        
        if response.status_code == 200:
            if "output" in result:
                print("\n✅ SUCCESS! Your endpoint is working!")
                print("\n📝 VIDEO CAPTION:")
                print("=" * 50)
                print(result['output']['caption'])
                print("=" * 50)
                
                metadata = result['output'].get('metadata', {})
                print(f"\n📊 Processing Stats:")
                print(f"  • Frames analyzed: {metadata.get('num_frames', 'N/A')}")
                print(f"  • FPS used: {metadata.get('fps_used', 'N/A')}")
                print(f"  • Processing time: {metadata.get('processing_time', 'N/A')}s")
            else:
                print("\n⚠️ Response received but unexpected format:")
                print(json.dumps(result, indent=2))
        elif response.status_code == 401:
            print("\n❌ Authentication failed!")
            print("You need to add your API key (see instructions above)")
        else:
            print(f"\n❌ Error (Status {response.status_code}):")
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(json.dumps(result, indent=2))
            
    except requests.exceptions.Timeout:
        print("\n⏱️ Request timed out (endpoint might be cold starting)")
        print("Try running again in a minute")
    except Exception as e:
        print(f"\n❌ Request failed: {e}")

if __name__ == "__main__":
    test_endpoint()
    
    print("\n" + "=" * 50)
    print("📌 Next steps:")
    print("1. Add your API key to this file")
    print("2. Run again: python quick_test.py")
    print("3. Try with your own video files")