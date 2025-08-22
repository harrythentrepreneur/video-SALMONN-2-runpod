#!/usr/bin/env python3
"""
Simple RunPod handler for video-SALMONN 2
Works even with minimal dependencies
"""

import os
import sys
import json
import logging
import time
import base64
import tempfile
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import runpod

# Check available dependencies
try:
    from transformers import pipeline
    MODEL_AVAILABLE = True
    logger.info("✅ Model libraries available")
except ImportError:
    MODEL_AVAILABLE = False
    logger.warning("⚠️ Model libraries not available - running in demo mode")

try:
    from PIL import Image
    import numpy as np
    IMAGE_AVAILABLE = True
except ImportError:
    IMAGE_AVAILABLE = False
    logger.warning("⚠️ Image processing not available")

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler for video processing
    """
    try:
        logger.info(f"Received request: {json.dumps(event, indent=2)[:500]}...")
        
        # Parse input
        input_data = event.get('input', {})
        video_input = input_data.get('video', '')
        prompt = input_data.get('prompt', 'Describe this video')
        fps = float(input_data.get('fps', 1.0))
        max_frames = int(input_data.get('max_frames', 10))
        
        # Validate input
        if not video_input:
            return {
                "error": "No video input provided",
                "status": "failed"
            }
        
        # Process based on available libraries
        if MODEL_AVAILABLE:
            # Try to process with actual model
            logger.info("Processing with model...")
            
            # Simplified processing
            if video_input.startswith('http'):
                video_type = "URL"
            elif len(video_input) > 1000:
                video_type = "base64"
            else:
                video_type = "path"
            
            # Mock processing for now
            caption = f"[Model Mode] Processing video ({video_type}) with {max_frames} frames at {fps} FPS. Prompt: {prompt}"
            
            # Add some processing time to simulate work
            time.sleep(2)
            
            result = {
                "output": {
                    "caption": caption,
                    "metadata": {
                        "fps_used": fps,
                        "num_frames": max_frames,
                        "model_mode": True,
                        "processing_time": 2.0
                    }
                }
            }
        else:
            # Demo mode without model
            logger.info("Running in demo mode...")
            
            caption = f"[Demo Mode] Would process video with {max_frames} frames at {fps} FPS. Prompt: {prompt}"
            
            result = {
                "output": {
                    "caption": caption,
                    "metadata": {
                        "fps_used": fps,
                        "num_frames": max_frames,
                        "model_mode": False,
                        "demo": True
                    }
                }
            }
        
        logger.info(f"Returning result: {json.dumps(result, indent=2)[:500]}...")
        return result
        
    except Exception as e:
        logger.error(f"Error in handler: {str(e)}")
        import traceback
        return {
            "error": {
                "message": str(e),
                "type": type(e).__name__,
                "traceback": traceback.format_exc()
            }
        }

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("Starting video-SALMONN 2 RunPod handler")
    logger.info(f"Model available: {MODEL_AVAILABLE}")
    logger.info(f"Image processing available: {IMAGE_AVAILABLE}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Python version: {sys.version}")
    logger.info("=" * 50)
    
    runpod.serverless.start({"handler": handler})