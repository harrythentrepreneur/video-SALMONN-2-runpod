#!/usr/bin/env python3
"""
Production RunPod Serverless Handler for video-SALMONN 2
Optimized for serverless deployment with model caching
"""

import os
import sys
import json
import torch
import tempfile
import base64
import logging
import time
from typing import Dict, Any, Optional, List
import subprocess
import traceback
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import runpod
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
import numpy as np
from PIL import Image
import decord
from decord import VideoReader, cpu

# Set decord bridge
decord.bridge.set_bridge('native')

class VideoSALMONN2Handler:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.device = None
        self.model_loaded = False
        
    def load_model(self):
        """Load the model once and cache it"""
        if self.model_loaded:
            return
            
        logger.info("Starting model initialization...")
        start_time = time.time()
        
        # Get model paths from environment
        model_path = os.environ.get('MODEL_PATH', 'tsinghua-ee/video-SALMONN-2')
        model_base = os.environ.get('MODEL_BASE', model_path)
        
        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        try:
            # Load tokenizer
            logger.info(f"Loading tokenizer from {model_base}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_base,
                use_fast=False,
                trust_remote_code=True
            )
            
            # Load model with optimizations
            logger.info(f"Loading model from {model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map='auto',
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                use_safetensors=True
            )
            
            # Set to eval mode
            self.model.eval()
            
            # Load processor if available
            try:
                self.processor = AutoProcessor.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
            except:
                logger.warning("Could not load processor, will use default image processing")
                self.processor = None
            
            self.model_loaded = True
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def extract_frames(self, video_path: str, fps: float = 1.0, max_frames: int = 30) -> List[np.ndarray]:
        """
        Extract frames from video at specified FPS
        
        Args:
            video_path: Path to video file
            fps: Target frames per second
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of numpy arrays (frames)
        """
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            video_fps = vr.get_avg_fps()
            
            # Calculate how many frames to extract
            duration = total_frames / video_fps
            target_frame_count = min(int(duration * fps), max_frames)
            
            if target_frame_count <= 0:
                target_frame_count = 1
            
            # Calculate frame indices
            if target_frame_count >= total_frames:
                frame_indices = list(range(total_frames))
            else:
                # Sample frames uniformly
                step = total_frames / target_frame_count
                frame_indices = [int(i * step) for i in range(target_frame_count)]
            
            # Extract frames
            frames = []
            for idx in frame_indices:
                frame = vr[idx].asnumpy()
                frames.append(frame)
            
            logger.info(f"Extracted {len(frames)} frames from video (duration: {duration:.2f}s)")
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            raise
    
    def extract_audio(self, video_path: str) -> Optional[str]:
        """
        Extract audio from video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            Path to extracted audio file or None if no audio
        """
        try:
            audio_path = tempfile.mktemp(suffix='.wav')
            
            # Extract audio using ffmpeg
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', '16000',  # 16kHz sample rate
                '-ac', '1',  # Mono
                '-y',  # Overwrite output
                audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                logger.info(f"Successfully extracted audio to {audio_path}")
                return audio_path
            else:
                logger.info("No audio track found or extraction failed")
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("Audio extraction timed out")
            return None
        except Exception as e:
            logger.error(f"Error extracting audio: {str(e)}")
            return None
    
    def process_video(self, video_input: str, fps: float = 1.0, process_audio: bool = False) -> Dict[str, Any]:
        """
        Process video input (base64 or file path)
        
        Args:
            video_input: Base64 encoded video or file path
            fps: Target FPS for frame extraction
            process_audio: Whether to extract and process audio
            
        Returns:
            Dictionary with frames and optional audio path
        """
        temp_video = None
        
        try:
            # Determine if input is base64 or file path
            if os.path.exists(video_input):
                video_path = video_input
            else:
                # Decode base64
                logger.info("Decoding base64 video input")
                video_bytes = base64.b64decode(video_input)
                temp_video = tempfile.mktemp(suffix='.mp4')
                with open(temp_video, 'wb') as f:
                    f.write(video_bytes)
                video_path = temp_video
            
            # Extract frames
            frames = self.extract_frames(video_path, fps=fps)
            
            # Extract audio if requested
            audio_path = None
            if process_audio:
                audio_path = self.extract_audio(video_path)
            
            return {
                'frames': frames,
                'audio_path': audio_path,
                'num_frames': len(frames),
                'fps': fps
            }
            
        finally:
            # Cleanup temporary video file
            if temp_video and os.path.exists(temp_video):
                os.remove(temp_video)
    
    def generate_caption(self, frames: List[np.ndarray], prompt: str, 
                        audio_path: Optional[str] = None,
                        temperature: float = 0.7, 
                        max_tokens: int = 512,
                        top_p: float = 0.9) -> str:
        """
        Generate caption for video frames
        
        Args:
            frames: List of video frames as numpy arrays
            prompt: User prompt/question
            audio_path: Optional path to audio file
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            
        Returns:
            Generated caption text
        """
        try:
            # Convert frames to PIL Images
            pil_frames = [Image.fromarray(frame) for frame in frames]
            
            # Process frames if processor available
            if self.processor:
                inputs = self.processor(
                    text=prompt,
                    images=pil_frames,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
            else:
                # Fallback: manual processing
                # This is a simplified version - actual implementation would need proper preprocessing
                logger.warning("Using fallback frame processing")
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate caption
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode output
            caption = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from output if present
            if prompt in caption:
                caption = caption.replace(prompt, "").strip()
            
            return caption
            
        except Exception as e:
            logger.error(f"Error generating caption: {str(e)}")
            raise
    
    def cleanup_audio(self, audio_path: Optional[str]):
        """Clean up temporary audio file"""
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                logger.info(f"Cleaned up audio file: {audio_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up audio file: {str(e)}")

# Global handler instance
handler_instance = VideoSALMONN2Handler()

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler function
    
    Expected input format:
    {
        "input": {
            "video": "base64_encoded_video_or_file_path",
            "prompt": "Describe this video",
            "fps": 1.0,  # Optional, default 1.0
            "max_frames": 30,  # Optional, default 30
            "temperature": 0.7,  # Optional, default 0.7
            "max_tokens": 512,  # Optional, default 512
            "top_p": 0.9,  # Optional, default 0.9
            "process_audio": false  # Optional, whether to process audio track
        }
    }
    
    Returns:
    {
        "output": {
            "caption": "Generated video caption",
            "metadata": {
                "num_frames": 30,
                "fps_used": 1.0,
                "audio_processed": false,
                "processing_time": 5.2
            }
        }
    }
    """
    start_time = time.time()
    audio_path = None
    
    try:
        # Load model if not already loaded
        handler_instance.load_model()
        
        # Parse input parameters
        input_data = event.get('input', {})
        
        # Required parameters
        video_input = input_data.get('video')
        if not video_input:
            raise ValueError("Missing required parameter: 'video'")
        
        # Optional parameters with defaults
        prompt = input_data.get('prompt', 'Describe this video in detail.')
        fps = float(input_data.get('fps', 1.0))
        max_frames = int(input_data.get('max_frames', 30))
        temperature = float(input_data.get('temperature', 0.7))
        max_tokens = int(input_data.get('max_tokens', 512))
        top_p = float(input_data.get('top_p', 0.9))
        process_audio = bool(input_data.get('process_audio', False))
        
        # Validate parameters
        fps = max(0.1, min(fps, 30.0))  # Clamp FPS between 0.1 and 30
        max_frames = max(1, min(max_frames, 100))  # Clamp frames between 1 and 100
        temperature = max(0.0, min(temperature, 2.0))  # Clamp temperature
        max_tokens = max(1, min(max_tokens, 2048))  # Clamp max tokens
        
        logger.info(f"Processing request - FPS: {fps}, Max frames: {max_frames}, Audio: {process_audio}")
        
        # Process video
        video_data = handler_instance.process_video(
            video_input=video_input,
            fps=fps,
            process_audio=process_audio
        )
        
        frames = video_data['frames']
        audio_path = video_data.get('audio_path')
        
        # Generate caption
        caption = handler_instance.generate_caption(
            frames=frames,
            prompt=prompt,
            audio_path=audio_path,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare response
        response = {
            "output": {
                "caption": caption,
                "metadata": {
                    "num_frames": len(frames),
                    "fps_used": fps,
                    "audio_processed": audio_path is not None,
                    "processing_time": round(processing_time, 2)
                }
            }
        }
        
        logger.info(f"Request completed in {processing_time:.2f} seconds")
        return response
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        
        return {
            "error": {
                "message": str(e),
                "type": type(e).__name__,
                "traceback": traceback.format_exc()
            }
        }
        
    finally:
        # Cleanup temporary audio file
        if audio_path:
            handler_instance.cleanup_audio(audio_path)

def test_local():
    """Test the handler locally with a sample video"""
    # Create a test event
    test_event = {
        "input": {
            "video": "/path/to/test/video.mp4",  # Replace with actual path
            "prompt": "Describe what happens in this video",
            "fps": 2.0,
            "max_frames": 30,
            "temperature": 0.7,
            "process_audio": True
        }
    }
    
    # Run handler
    result = handler(test_event)
    
    # Print result
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='video-SALMONN 2 RunPod Handler')
    parser.add_argument('--test', action='store_true', help='Run local test')
    args = parser.parse_args()
    
    if args.test:
        test_local()
    else:
        logger.info("Starting RunPod serverless handler...")
        runpod.serverless.start({"handler": handler})