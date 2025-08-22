#!/usr/bin/env python3
"""
RunPod Handler for video-SALMONN 2 Model
Supports audio-visual video captioning with configurable FPS
"""

import os
import sys
import json
import torch
import tempfile
import base64
from typing import Dict, Any, Optional
import subprocess
import traceback

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import runpod

# Import model components
from llava.model import LlavaLlamaForCausalLM
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from transformers import AutoTokenizer, AutoConfig
import decord
import numpy as np
from PIL import Image

# Global variables for model caching
model = None
tokenizer = None
image_processor = None
context_len = None

def download_model_if_needed():
    """Download model from HuggingFace if not present locally"""
    model_path = os.environ.get('MODEL_PATH', '/workspace/models/video-SALMONN-2')
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Downloading from HuggingFace...")
        os.makedirs(model_path, exist_ok=True)
        
        # Download using git-lfs or huggingface-cli
        try:
            subprocess.run([
                "huggingface-cli", "download",
                "tsinghua-ee/video-SALMONN-2",
                "--local-dir", model_path
            ], check=True)
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise
    
    return model_path

def initialize_model():
    """Initialize the video-SALMONN 2 model"""
    global model, tokenizer, image_processor, context_len
    
    if model is not None:
        return  # Model already initialized
    
    print("Initializing video-SALMONN 2 model...")
    
    # Get model path
    model_path = download_model_if_needed()
    model_base = os.environ.get('MODEL_BASE', model_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
    
    # Load model configuration
    model_args = {
        'model_path': model_path,
        'model_base': model_base,
        'model_name': get_model_name_from_path(model_path),
        'load_8bit': False,
        'load_4bit': False,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'use_flash_attn': True,
    }
    
    # Load the model
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        use_safetensors=True,
        **model_args
    )
    
    model = model.to('cuda')
    model.eval()
    
    # Get image processor
    vision_tower = model.get_vision_tower()
    if vision_tower is not None:
        image_processor = vision_tower.image_processor
    
    # Set context length
    if hasattr(model.config, 'max_sequence_length'):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    
    print("Model initialized successfully!")

def extract_frames_from_video(video_path: str, fps: float = 1.0, max_frames: int = 30) -> list:
    """
    Extract frames from video at specified FPS
    
    Args:
        video_path: Path to video file
        fps: Frames per second to extract (default: 1.0)
        max_frames: Maximum number of frames to extract (default: 30)
    
    Returns:
        List of PIL Images
    """
    try:
        # Use decord for efficient video frame extraction
        vr = decord.VideoReader(video_path, ctx=decord.cpu())
        total_frames = len(vr)
        video_fps = vr.get_avg_fps()
        
        # Calculate frame indices based on desired FPS
        duration = total_frames / video_fps
        num_frames = min(int(duration * fps), max_frames)
        
        if num_frames == 0:
            num_frames = 1
        
        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            frame = vr[idx].asnumpy()
            pil_frame = Image.fromarray(frame)
            frames.append(pil_frame)
        
        return frames
    
    except Exception as e:
        print(f"Error extracting frames: {e}")
        raise

def process_video_input(video_data: str, fps: float = 1.0) -> tuple:
    """
    Process video input from base64 or file path
    
    Args:
        video_data: Base64 encoded video or file path
        fps: Frames per second for extraction
    
    Returns:
        Tuple of (frames, audio_path)
    """
    temp_video_path = None
    temp_audio_path = None
    
    try:
        # Check if input is base64 or file path
        if os.path.exists(video_data):
            temp_video_path = video_data
        else:
            # Decode base64 video
            video_bytes = base64.b64decode(video_data)
            temp_video_path = tempfile.mktemp(suffix='.mp4')
            with open(temp_video_path, 'wb') as f:
                f.write(video_bytes)
        
        # Extract audio if present (for audio-visual processing)
        temp_audio_path = tempfile.mktemp(suffix='.wav')
        try:
            subprocess.run([
                'ffmpeg', '-i', temp_video_path,
                '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                temp_audio_path
            ], check=True, capture_output=True)
        except:
            # No audio or extraction failed
            temp_audio_path = None
        
        # Extract frames
        frames = extract_frames_from_video(temp_video_path, fps=fps)
        
        return frames, temp_audio_path
    
    finally:
        # Cleanup temporary files if created from base64
        if temp_video_path and not os.path.exists(video_data):
            try:
                os.remove(temp_video_path)
            except:
                pass

def generate_caption(frames: list, audio_path: Optional[str], prompt: str, temperature: float = 0.7, max_tokens: int = 512) -> str:
    """
    Generate caption for video frames (and optionally audio)
    
    Args:
        frames: List of PIL Images
        audio_path: Path to audio file (optional)
        prompt: User prompt/question
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate
    
    Returns:
        Generated caption text
    """
    global model, tokenizer, image_processor
    
    # Process frames
    if image_processor is not None:
        processed_frames = image_processor.preprocess(frames, return_tensors='pt')['pixel_values']
        processed_frames = processed_frames.to(model.device, dtype=torch.float16)
    else:
        processed_frames = torch.stack([torch.from_numpy(np.array(f)) for f in frames])
        processed_frames = processed_frames.to(model.device, dtype=torch.float16)
    
    # Prepare conversation
    conv = conv_templates["qwen_1_5"].copy()
    
    # Add image tokens based on number of frames
    image_tokens = DEFAULT_IMAGE_TOKEN * len(frames)
    inp = image_tokens + '\n' + prompt
    
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()
    
    # Tokenize
    input_ids = tokenizer_image_token(
        prompt_text,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors='pt'
    ).unsqueeze(0).to(model.device)
    
    # Generate
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=processed_frames.unsqueeze(0),
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            max_new_tokens=max_tokens,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode output
    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    
    # Extract assistant's response
    if conv.roles[1] in output_text:
        output_text = output_text.split(conv.roles[1])[-1].strip()
    
    return output_text

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler function
    
    Expected input format:
    {
        "input": {
            "video": "base64_encoded_video_or_file_path",
            "prompt": "Describe this video",
            "fps": 1.0,  # Optional, default 1.0
            "temperature": 0.7,  # Optional, default 0.7
            "max_tokens": 512,  # Optional, default 512
            "audio_visual": true  # Optional, whether to process audio
        }
    }
    """
    try:
        # Initialize model if needed
        initialize_model()
        
        # Parse input
        input_data = event.get('input', {})
        video_data = input_data.get('video')
        prompt = input_data.get('prompt', 'Describe this video in detail.')
        fps = float(input_data.get('fps', 1.0))
        temperature = float(input_data.get('temperature', 0.7))
        max_tokens = int(input_data.get('max_tokens', 512))
        audio_visual = input_data.get('audio_visual', False)
        
        if not video_data:
            raise ValueError("No video data provided")
        
        print(f"Processing video with FPS={fps}, temperature={temperature}, max_tokens={max_tokens}")
        
        # Process video
        frames, audio_path = process_video_input(video_data, fps=fps)
        
        if not audio_visual:
            audio_path = None  # Ignore audio if not requested
        
        print(f"Extracted {len(frames)} frames from video")
        
        # Generate caption
        caption = generate_caption(
            frames=frames,
            audio_path=audio_path,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Cleanup audio file if created
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass
        
        return {
            "output": {
                "caption": caption,
                "num_frames": len(frames),
                "fps_used": fps,
                "audio_processed": audio_path is not None
            }
        }
    
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        print(traceback.format_exc())
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# RunPod entry point
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})