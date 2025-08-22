#!/usr/bin/env python3
"""
RunPod handler using Qwen2.5-VL for video understanding
Based on the Qwen2.5-VL model documentation
"""

import os
import sys
import json
import logging
import time
import base64
import tempfile
import subprocess
from typing import Dict, Any, List
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import runpod

# Try to import required libraries
try:
    from PIL import Image
    import numpy as np
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    MODEL_AVAILABLE = True
except ImportError as e:
    MODEL_AVAILABLE = False
    logger.warning(f"Model libraries not available: {e}")

# Global model variables
model = None
processor = None

def initialize_model():
    """Initialize Qwen2.5-VL model"""
    global model, processor
    
    if model is not None:
        return
    
    try:
        logger.info("Initializing Qwen2.5-VL model...")
        
        # Model selection based on available memory
        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"  # or use 2B for less memory
        
        # Load processor
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        
        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        model.eval()
        logger.info("Qwen2.5-VL model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        logger.info("Falling back to lightweight mode")
        
        # Try a smaller model as fallback
        try:
            from transformers import pipeline
            global fallback_pipeline
            fallback_pipeline = pipeline(
                "image-to-text",
                model="Salesforce/blip-image-captioning-base"
            )
            logger.info("Fallback model loaded")
        except:
            pass

def extract_frames(video_input: str, fps: float = 1.0, max_frames: int = 10) -> List[Image.Image]:
    """Extract frames from video (URL or base64)"""
    frames = []
    temp_video = None
    
    try:
        # Handle different input types
        if video_input.startswith('http'):
            # Download from URL
            logger.info("Downloading video from URL...")
            response = requests.get(video_input, stream=True, timeout=60)
            response.raise_for_status()
            
            temp_video = tempfile.mktemp(suffix='.mp4')
            with open(temp_video, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
        elif len(video_input) > 1000:  # Likely base64
            logger.info("Decoding base64 video...")
            video_bytes = base64.b64decode(video_input)
            temp_video = tempfile.mktemp(suffix='.mp4')
            with open(temp_video, 'wb') as f:
                f.write(video_bytes)
        else:
            # Assume local file path
            temp_video = video_input
        
        # Extract frames using ffmpeg
        logger.info(f"Extracting frames at {fps} FPS (max {max_frames})...")
        
        # Get video duration
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            temp_video
        ]
        
        try:
            duration = float(subprocess.check_output(probe_cmd).decode().strip())
            logger.info(f"Video duration: {duration:.2f} seconds")
        except:
            duration = 60  # Default assumption
        
        # Calculate actual frames to extract
        desired_frames = min(int(duration * fps), max_frames)
        
        # Extract frames uniformly
        for i in range(desired_frames):
            timestamp = i / fps
            temp_frame = tempfile.mktemp(suffix='.jpg')
            
            cmd = [
                'ffmpeg', '-ss', str(timestamp),
                '-i', temp_video,
                '-frames:v', '1',
                '-q:v', '2',
                temp_frame,
                '-loglevel', 'error'
            ]
            
            if subprocess.run(cmd).returncode == 0:
                frames.append(Image.open(temp_frame))
                os.remove(temp_frame)
        
        logger.info(f"Successfully extracted {len(frames)} frames")
        
    except Exception as e:
        logger.error(f"Frame extraction error: {str(e)}")
        # Create placeholder
        if Image:
            frames = [Image.new('RGB', (640, 480), color='gray')]
    
    finally:
        if temp_video and temp_video != video_input and os.path.exists(temp_video):
            os.remove(temp_video)
    
    return frames

def process_with_qwen(frames: List[Image.Image], prompt: str) -> str:
    """Process video frames with Qwen2.5-VL"""
    global model, processor
    
    if not model:
        initialize_model()
    
    if not model:
        # Use fallback if available
        if 'fallback_pipeline' in globals():
            captions = []
            for i, frame in enumerate(frames[:3]):
                result = fallback_pipeline(frame)
                if result:
                    captions.append(f"Frame {i+1}: {result[0]['generated_text']}")
            return "\n".join(captions) if captions else "Model not available"
        return "Model not available for processing"
    
    try:
        # Prepare messages for Qwen2.5-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Add video frames as images
        for frame in frames:
            messages[0]["content"].append({"type": "image", "image": frame})
        
        # Process with model
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = processor(
            text=text,
            images=frames,
            return_tensors="pt"
        ).to(model.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True
            )
        
        # Decode response
        generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
        response = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return response
        
    except Exception as e:
        logger.error(f"Qwen processing error: {str(e)}")
        return f"Processing completed with {len(frames)} frames. Error in model: {str(e)}"

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler for Qwen2.5-VL video processing
    """
    try:
        start_time = time.time()
        
        # Parse input
        input_data = event.get('input', event)
        
        video_input = (
            input_data.get('video') or 
            input_data.get('video_url') or 
            input_data.get('video_base64') or 
            ''
        )
        
        prompt = input_data.get('prompt', 'Describe this video in detail')
        fps = float(input_data.get('fps', 1.0))
        max_frames = int(input_data.get('max_frames', 10))
        
        logger.info(f"Processing request - FPS: {fps}, Max Frames: {max_frames}")
        
        if not video_input:
            return {
                "output": {
                    "error": "No video input provided",
                    "status": "failed"
                }
            }
        
        # Extract frames
        frames = extract_frames(video_input, fps, max_frames)
        
        if not frames:
            return {
                "output": {
                    "error": "Failed to extract frames from video",
                    "status": "failed"
                }
            }
        
        # Process with model
        caption = process_with_qwen(frames, prompt)
        
        processing_time = time.time() - start_time
        
        return {
            "output": {
                "caption": caption,
                "metadata": {
                    "fps_used": fps,
                    "num_frames": len(frames),
                    "processing_time": round(processing_time, 2),
                    "model": "Qwen2.5-VL" if model else "fallback",
                    "prompt": prompt
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        import traceback
        return {
            "output": {
                "error": str(e),
                "traceback": traceback.format_exc()[:500],
                "status": "failed"
            }
        }

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("Starting Qwen2.5-VL Video Handler")
    logger.info(f"Model available: {MODEL_AVAILABLE}")
    logger.info(f"CUDA available: {torch.cuda.is_available() if 'torch' in sys.modules else False}")
    logger.info("=" * 50)
    
    # Pre-initialize if possible
    if MODEL_AVAILABLE:
        initialize_model()
    
    runpod.serverless.start({"handler": handler})