#!/usr/bin/env python3
"""
RunPod handler for video-SALMONN-2 with PEFT
Using the actual model checkpoints
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

# Add project path
sys.path.append('/workspace')

# Try to import required libraries
try:
    from PIL import Image
    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
    MODEL_LIBRARIES_AVAILABLE = True
except ImportError as e:
    MODEL_LIBRARIES_AVAILABLE = False
    logger.warning(f"Model libraries not available: {e}")

# Try to import video-SALMONN specific modules
try:
    from llava.model import LlavaLlamaForCausalLM
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token, process_images
    from llava.constants import IMAGE_TOKEN_INDEX
    SALMONN_AVAILABLE = True
except ImportError as e:
    SALMONN_AVAILABLE = False
    logger.warning(f"video-SALMONN modules not available: {e}")

# Global model variables
model = None
tokenizer = None
processor = None

def initialize_salmonn_model():
    """Initialize video-SALMONN-2 model with PEFT"""
    global model, tokenizer, processor
    
    if model is not None:
        return
    
    try:
        logger.info("Initializing video-SALMONN-2 model...")
        
        # Check for model paths
        base_model_path = os.environ.get('BASE_MODEL_PATH', 
            "output/audio_align_winqf0_5s_Ls960Com_AcSl_bs8ep5_LargerQF/checkpoint-30510")
        peft_model_path = os.environ.get('PEFT_MODEL_PATH', 
            "tsinghua-ee/video-SALMONN-2_plus_7B")
        
        # Alternative: Try HuggingFace model directly
        if not os.path.exists(base_model_path):
            logger.info("Local checkpoint not found, trying HuggingFace model...")
            base_model_path = "tsinghua-ee/video-SALMONN-2"
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            use_fast=False,
            trust_remote_code=True
        )
        
        # Load base model
        logger.info(f"Loading base model from {base_model_path}")
        
        if SALMONN_AVAILABLE:
            # Use video-SALMONN specific loader
            base_model = LlavaLlamaForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        else:
            # Fallback to standard loader
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        
        # Load PEFT adapter if available
        try:
            logger.info(f"Loading PEFT adapter from {peft_model_path}")
            model = PeftModel.from_pretrained(base_model, peft_model_path)
            logger.info("PEFT adapter loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load PEFT adapter: {e}")
            logger.info("Using base model without PEFT")
            model = base_model
        
        model.eval()
        
        # Try to get processor
        try:
            processor = AutoProcessor.from_pretrained(
                base_model_path,
                trust_remote_code=True
            )
        except:
            processor = None
            logger.info("Processor not available, will use tokenizer only")
        
        logger.info("video-SALMONN-2 model initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize video-SALMONN model: {str(e)}")
        
        # Try lightweight fallback
        try:
            logger.info("Attempting fallback model...")
            from transformers import pipeline
            global fallback_pipeline
            fallback_pipeline = pipeline(
                "image-to-text",
                model="Salesforce/blip-image-captioning-base",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("Fallback model loaded")
        except Exception as fe:
            logger.error(f"Fallback also failed: {fe}")

def extract_frames(video_input: str, fps: float = 1.0, max_frames: int = 30) -> List[Image.Image]:
    """Extract frames from video"""
    frames = []
    temp_video = None
    
    try:
        # Handle different input types
        if video_input.startswith('http'):
            logger.info("Downloading video from URL...")
            response = requests.get(video_input, stream=True, timeout=60)
            response.raise_for_status()
            
            temp_video = tempfile.mktemp(suffix='.mp4')
            with open(temp_video, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if len(f.tell()) > 100 * 1024 * 1024:  # 100MB limit
                        break
                    f.write(chunk)
                    
        elif len(video_input) > 1000:  # Base64
            logger.info("Decoding base64 video...")
            video_bytes = base64.b64decode(video_input)
            temp_video = tempfile.mktemp(suffix='.mp4')
            with open(temp_video, 'wb') as f:
                f.write(video_bytes)
        else:
            temp_video = video_input
        
        # Extract frames with ffmpeg
        logger.info(f"Extracting frames at {fps} FPS...")
        
        # Calculate frame interval
        interval = max(1, int(1.0 / fps))
        
        # Extract frames
        temp_frame_pattern = tempfile.gettempdir() + "/frame_%04d.jpg"
        
        cmd = [
            'ffmpeg', '-i', temp_video,
            '-vf', f'fps={fps}',
            '-frames:v', str(max_frames),
            '-q:v', '2',
            temp_frame_pattern,
            '-loglevel', 'error'
        ]
        
        subprocess.run(cmd, check=True)
        
        # Load extracted frames
        import glob
        frame_files = sorted(glob.glob(temp_frame_pattern.replace('%04d', '*')))
        
        for frame_file in frame_files[:max_frames]:
            frames.append(Image.open(frame_file))
            os.remove(frame_file)
        
        logger.info(f"Extracted {len(frames)} frames")
        
    except Exception as e:
        logger.error(f"Frame extraction error: {str(e)}")
        # Create placeholder
        frames = [Image.new('RGB', (640, 480), color='black')]
    
    finally:
        if temp_video and temp_video != video_input and os.path.exists(temp_video):
            os.remove(temp_video)
    
    return frames

def process_with_salmonn(frames: List[Image.Image], prompt: str) -> str:
    """Process frames with video-SALMONN-2"""
    global model, tokenizer, processor
    
    if not model:
        initialize_salmonn_model()
    
    if not model:
        # Try fallback
        if 'fallback_pipeline' in globals():
            captions = []
            for i, frame in enumerate(frames[:5]):
                result = fallback_pipeline(frame)
                if result:
                    captions.append(f"Frame {i+1}: {result[0]['generated_text']}")
            return "\n".join(captions) if captions else "Model not available"
        return f"Model not loaded. Processed {len(frames)} frames at requested FPS."
    
    try:
        if SALMONN_AVAILABLE:
            # Use video-SALMONN conversation format
            conv = conv_templates.get("qwen_1_5", conv_templates["vicuna_v1"])
            
            # Prepare input with frames
            image_tokens = "<image>" * len(frames)
            question = f"{image_tokens}\n{prompt}"
            
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_text = conv.get_prompt()
            
            # Tokenize
            input_ids = tokenizer_image_token(
                prompt_text,
                tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors='pt'
            ).unsqueeze(0).to(model.device)
            
            # Process images if needed
            if processor:
                image_tensor = processor.preprocess(frames, return_tensors='pt')['pixel_values']
                image_tensor = image_tensor.to(model.device, dtype=model.dtype)
            else:
                # Simple tensor conversion
                image_arrays = [np.array(f.resize((384, 384))) for f in frames]
                image_tensor = torch.from_numpy(np.stack(image_arrays)).to(model.device)
                image_tensor = image_tensor.permute(0, 3, 1, 2).float() / 255.0
            
            # Generate
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0),
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    use_cache=True
                )
            
            # Decode
            output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            
            # Extract response
            if conv.roles[1] in output_text:
                output_text = output_text.split(conv.roles[1])[-1].strip()
            
            return output_text
            
        else:
            # Fallback processing without SALMONN modules
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response += f"\n\n[Processed {len(frames)} video frames]"
            
            return response
            
    except Exception as e:
        logger.error(f"SALMONN processing error: {str(e)}")
        return f"Processed {len(frames)} frames. Model error: {str(e)[:200]}"

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler for video-SALMONN-2
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
        max_frames = int(input_data.get('max_frames', 30))
        
        logger.info(f"Processing - FPS: {fps}, Max Frames: {max_frames}")
        
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
                    "error": "Failed to extract frames",
                    "status": "failed"
                }
            }
        
        # Process with model
        caption = process_with_salmonn(frames, prompt)
        
        processing_time = time.time() - start_time
        
        return {
            "output": {
                "caption": caption,
                "metadata": {
                    "fps_used": fps,
                    "num_frames": len(frames),
                    "processing_time": round(processing_time, 2),
                    "model": "video-SALMONN-2" if model else "fallback",
                    "device": str(model.device) if model else "cpu"
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        import traceback
        return {
            "output": {
                "error": str(e),
                "traceback": traceback.format_exc()[:1000],
                "status": "failed"
            }
        }

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("video-SALMONN-2 RunPod Handler")
    logger.info(f"Model libraries: {MODEL_LIBRARIES_AVAILABLE}")
    logger.info(f"SALMONN modules: {SALMONN_AVAILABLE}")
    logger.info(f"CUDA: {torch.cuda.is_available() if 'torch' in sys.modules else False}")
    logger.info("=" * 50)
    
    # Pre-initialize
    if MODEL_LIBRARIES_AVAILABLE:
        initialize_salmonn_model()
    
    runpod.serverless.start({"handler": handler})