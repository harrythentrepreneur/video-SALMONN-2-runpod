#!/bin/bash

echo "🚀 Building and pushing video-SALMONN-2 to Docker Hub"
echo "=================================================="

# Docker Hub username - CHANGE THIS TO YOUR USERNAME
DOCKER_USERNAME="YOUR_DOCKERHUB_USERNAME"

if [ "$DOCKER_USERNAME" = "YOUR_DOCKERHUB_USERNAME" ]; then
    echo "❌ Please edit this file and set your Docker Hub username!"
    echo "   Edit line 6: DOCKER_USERNAME=\"your-actual-username\""
    exit 1
fi

echo "📦 Step 1: Building Docker image locally..."
docker build -t video-salmonn2:latest .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed!"
    echo "Make sure Docker Desktop is running"
    exit 1
fi

echo "✅ Build complete!"

echo "📝 Step 2: Tagging image for Docker Hub..."
docker tag video-salmonn2:latest $DOCKER_USERNAME/video-salmonn2:latest

echo "🔐 Step 3: Logging into Docker Hub..."
docker login

if [ $? -ne 0 ]; then
    echo "❌ Docker login failed!"
    echo "Please check your credentials"
    exit 1
fi

echo "⬆️ Step 4: Pushing to Docker Hub..."
echo "This may take several minutes..."
docker push $DOCKER_USERNAME/video-salmonn2:latest

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ SUCCESS! Image pushed to Docker Hub!"
    echo "=================================================="
    echo "📌 Your image URL: $DOCKER_USERNAME/video-salmonn2:latest"
    echo ""
    echo "Next steps:"
    echo "1. Go to RunPod: https://www.runpod.io/console/serverless"
    echo "2. Click on your endpoint"
    echo "3. Click 'Edit Endpoint'"
    echo "4. Change Container Image to: $DOCKER_USERNAME/video-salmonn2:latest"
    echo "5. Click 'Update'"
else
    echo "❌ Push failed! Check your internet connection and try again"
fi