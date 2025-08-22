# RunPod GitHub Integration Setup

## ✅ Complete Setup Guide

### 1. Repository Setup (DONE ✅)
- Your repo: https://github.com/harrythentrepreneur/video-SALMONN-2-runpod
- Dockerfile is configured for RunPod
- API keys are secured in .env file

### 2. Configure RunPod Endpoint

Go to your RunPod endpoint: https://www.runpod.io/console/serverless

#### Update Endpoint Settings:
1. Click on your endpoint `uyrc9kfx7k6rfr`
2. Click **"Edit Endpoint"**
3. Update these settings:

**Container Configuration:**
- **Container Image Source**: Select "GitHub"
- **GitHub URL**: `https://github.com/harrythentrepreneur/video-SALMONN-2-runpod`
- **Branch**: `main`
- **Dockerfile Path**: `Dockerfile` (root directory)

**Environment Variables:**
```
MODEL_PATH=tsinghua-ee/video-SALMONN-2
CUDA_VISIBLE_DEVICES=0
PYTHONUNBUFFERED=1
```

**GPU Configuration:**
- GPU Type: RTX 4090 or A100
- Container Disk: 30 GB
- Min Workers: 0
- Max Workers: 3

4. Click **"Update"**

### 3. Wait for Build

RunPod will now:
1. Pull your GitHub repo
2. Build the Docker image
3. Deploy it to your endpoint

This takes about 5-10 minutes. You can monitor progress in the "Logs" tab.

### 4. Test Your Endpoint

Once deployed, test it:

```bash
# Install python-dotenv if needed
pip install python-dotenv

# Test with sample video
python test_endpoint.py

# Test with your own video
python test_endpoint.py your_video.mp4
```

## 📝 Using Your API

### With Environment Variables (.env file)
```python
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv('RUNPOD_API_KEY')
ENDPOINT_ID = os.getenv('RUNPOD_ENDPOINT_ID')
```

### Configurable FPS
```python
payload = {
    "input": {
        "video": "your_video.mp4",
        "prompt": "Describe this video",
        "fps": 2.0,  # Adjust from 0.1 to 30
        "max_frames": 30
    }
}
```

## 🔧 Troubleshooting

### If build fails:
1. Check RunPod logs for errors
2. Ensure Dockerfile syntax is correct
3. Verify all dependencies in requirements.txt

### If endpoint doesn't respond:
1. Check if build completed successfully
2. Verify environment variables are set
3. Ensure GPU has enough memory

### To trigger rebuild:
```bash
# Make any small change
echo "# Rebuild trigger" >> README.md
git add . && git commit -m "Trigger rebuild"
git push origin main
```

## 🚀 Alternative: Docker Hub

If GitHub build has issues, use Docker Hub:

```bash
# Build and push to Docker Hub
docker build -t YOUR_USERNAME/video-salmonn2:latest .
docker push YOUR_USERNAME/video-salmonn2:latest

# Then in RunPod, use:
# Container Image: YOUR_USERNAME/video-salmonn2:latest
```

## 📊 FPS Settings Guide

| Video Type | Recommended FPS | Use Case |
|------------|----------------|----------|
| Long videos (>5 min) | 0.5-1.0 | Quick overview |
| Regular videos | 1.0-2.0 | Balanced analysis |
| Short clips | 2.0-5.0 | Detailed analysis |
| Action scenes | 5.0-10.0 | Motion tracking |

## 🔐 Security Notes

- Never commit .env file (it's in .gitignore)
- Use .env.example as template
- Rotate API keys regularly
- Keep credentials out of code