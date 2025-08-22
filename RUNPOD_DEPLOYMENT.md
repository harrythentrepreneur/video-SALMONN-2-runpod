# video-SALMONN 2 RunPod Deployment Guide

This guide will help you deploy the video-SALMONN 2 model on RunPod for serverless inference with configurable FPS settings.

## Features

- **Configurable FPS**: Choose frames per second (0.1 to 30 FPS) for video processing
- **Audio-Visual Processing**: Optional audio track processing for enhanced captioning
- **Serverless Architecture**: Automatic scaling and pay-per-use pricing
- **Optimized Performance**: GPU acceleration with model caching
- **Base64 Support**: Accept video input as base64 or file paths

## Prerequisites

1. RunPod account with credits
2. Docker installed locally (for building custom image)
3. Access to GPU instances (recommended: A100, A6000, or RTX 4090)

## Quick Start

### Option 1: Deploy Pre-built Image (Recommended)

1. **Login to RunPod Console**
   - Navigate to [RunPod Serverless](https://www.runpod.io/console/serverless)

2. **Create New Endpoint**
   - Click "New Endpoint"
   - Select "Custom Container"
   - Container Image: `tsinghua-ee/video-salmonn2-runpod:latest` (if available)
   - GPU Type: Select based on needs (A100 40GB recommended)
   - Min Workers: 0
   - Max Workers: Set based on expected load

3. **Configure Environment Variables**
   ```
   MODEL_PATH=/workspace/models/video-SALMONN-2
   MODEL_BASE=/workspace/models/video-SALMONN-2
   CUDA_VISIBLE_DEVICES=0
   ```

### Option 2: Build and Deploy Custom Image

1. **Clone Repository**
   ```bash
   git clone https://github.com/bytedance/video-SALMONN-2.git
   cd video-SALMONN-2
   ```

2. **Build Docker Image**
   ```bash
   docker build -t video-salmonn2-runpod:latest .
   ```

3. **Push to Docker Registry**
   ```bash
   # Tag for your registry (e.g., Docker Hub)
   docker tag video-salmonn2-runpod:latest YOUR_USERNAME/video-salmonn2-runpod:latest
   docker push YOUR_USERNAME/video-salmonn2-runpod:latest
   ```

4. **Deploy on RunPod**
   - Use your custom image URL in RunPod endpoint configuration

## API Usage

### Request Format

Send POST requests to your RunPod endpoint:

```json
{
  "input": {
    "video": "base64_encoded_video_string_or_file_path",
    "prompt": "Describe what happens in this video",
    "fps": 2.0,
    "max_frames": 30,
    "temperature": 0.7,
    "max_tokens": 512,
    "top_p": 0.9,
    "process_audio": true
  }
}
```

### Parameters

| Parameter | Type | Default | Description | Range |
|-----------|------|---------|-------------|-------|
| `video` | string | Required | Base64 encoded video or file path | - |
| `prompt` | string | "Describe this video in detail." | Question or instruction for the model | - |
| `fps` | float | 1.0 | Frames per second to extract | 0.1 - 30.0 |
| `max_frames` | int | 30 | Maximum frames to process | 1 - 100 |
| `temperature` | float | 0.7 | Sampling temperature | 0.0 - 2.0 |
| `max_tokens` | int | 512 | Maximum tokens to generate | 1 - 2048 |
| `top_p` | float | 0.9 | Top-p sampling parameter | 0.0 - 1.0 |
| `process_audio` | bool | false | Process audio track if available | true/false |

### Response Format

```json
{
  "output": {
    "caption": "Generated video caption describing the content...",
    "metadata": {
      "num_frames": 30,
      "fps_used": 2.0,
      "audio_processed": true,
      "processing_time": 5.42
    }
  }
}
```

### Error Response

```json
{
  "error": {
    "message": "Error description",
    "type": "ErrorType",
    "traceback": "Full stack trace..."
  }
}
```

## Example Usage

### Python Client

```python
import requests
import base64
import json

# Your RunPod endpoint URL
ENDPOINT_URL = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync"
API_KEY = "YOUR_RUNPOD_API_KEY"

def process_video(video_path, prompt, fps=1.0):
    # Read and encode video
    with open(video_path, "rb") as f:
        video_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    # Prepare request
    payload = {
        "input": {
            "video": video_base64,
            "prompt": prompt,
            "fps": fps,
            "max_frames": 30,
            "temperature": 0.7,
            "process_audio": True
        }
    }
    
    # Send request
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(ENDPOINT_URL, json=payload, headers=headers)
    result = response.json()
    
    if "output" in result:
        print(f"Caption: {result['output']['caption']}")
        print(f"Processed {result['output']['metadata']['num_frames']} frames at {result['output']['metadata']['fps_used']} FPS")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    return result

# Example usage
result = process_video(
    video_path="sample_video.mp4",
    prompt="What is happening in this video? Describe the actions and any dialogue.",
    fps=2.0  # Extract 2 frames per second
)
```

### cURL Example

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "video": "'"$(base64 -i video.mp4)"'",
      "prompt": "Describe this video",
      "fps": 1.0
    }
  }'
```

## FPS Configuration Guide

Choose the appropriate FPS based on your use case:

| Use Case | Recommended FPS | Max Frames | Notes |
|----------|----------------|------------|-------|
| Quick Overview | 0.5 - 1.0 | 10-20 | Fast processing, general understanding |
| Action Videos | 2.0 - 5.0 | 30-50 | Captures motion and transitions |
| Detailed Analysis | 5.0 - 10.0 | 50-100 | Comprehensive frame coverage |
| Short Clips (<10s) | 10.0 - 30.0 | 30-100 | Near frame-by-frame analysis |

### Performance Considerations

- **Lower FPS (0.5-2.0)**: Faster processing, lower costs, suitable for long videos
- **Medium FPS (2.0-5.0)**: Balanced quality and speed, good for most use cases
- **Higher FPS (5.0+)**: More detailed analysis, higher costs, longer processing time

## Local Testing

Test the handler locally before deployment:

```bash
# Install requirements
pip install -r requirements.txt
pip install runpod

# Test with sample video
python runpod_serverless.py --test
```

## Monitoring and Debugging

### View Logs
- Access logs through RunPod dashboard
- Check endpoint metrics for performance monitoring

### Common Issues

1. **Out of Memory**
   - Reduce `max_frames` or `fps`
   - Use larger GPU instance

2. **Slow Processing**
   - Model loads on first request (cold start)
   - Consider keeping minimum workers > 0

3. **Video Format Issues**
   - Ensure video is in supported format (MP4, AVI, MOV)
   - Check video isn't corrupted

## Cost Optimization

1. **Adjust Worker Settings**
   - Set Min Workers = 0 for low traffic
   - Increase Max Workers during peak times

2. **Optimize FPS Settings**
   - Use lowest FPS that meets quality requirements
   - Cache results for repeated queries

3. **Choose Right GPU**
   - RTX 4090: Best cost/performance for most cases
   - A100: For high throughput requirements
   - A6000: Balance of memory and performance

## Advanced Configuration

### Custom Model Weights

To use different model checkpoints:

1. Download weights to `/workspace/models/`
2. Update environment variables:
   ```
   MODEL_PATH=/workspace/models/your-model
   MODEL_BASE=/workspace/models/your-base-model
   ```

### Batch Processing

For multiple videos, consider:
- Implementing queue system
- Using RunPod's async endpoints
- Batching frames from multiple videos

## Support

- **RunPod Issues**: Contact RunPod support
- **Model Issues**: Open issue on [GitHub](https://github.com/bytedance/video-SALMONN-2)
- **API Questions**: Check RunPod documentation

## License

This deployment follows the Apache 2.0 License of the video-SALMONN 2 project.