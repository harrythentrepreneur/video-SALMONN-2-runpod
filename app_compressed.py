#!/usr/bin/env python3
"""
Flask web interface with video compression for large files
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import base64
import requests
import json
from dotenv import load_dotenv
import tempfile
import subprocess
from werkzeug.utils import secure_filename

# Load environment variables
load_dotenv()

app = Flask(__name__, template_folder='templates')
CORS(app)

# Configuration
ENDPOINT_ID = os.getenv('RUNPOD_ENDPOINT_ID')
API_KEY = os.getenv('RUNPOD_API_KEY')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'webm', 'mkv'}

# Increase limits for large files
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def compress_video(input_path, max_size_mb=10):
    """Compress video to reduce size for API transmission"""
    output_path = tempfile.mktemp(suffix='.mp4')
    
    try:
        # Get input file size
        input_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
        print(f"Original video size: {input_size:.2f} MB")
        
        if input_size <= max_size_mb:
            # No compression needed
            with open(input_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        
        # Calculate compression ratio
        compression_ratio = max_size_mb / input_size
        
        # Compress video using ffmpeg
        # Reduce resolution and bitrate based on compression needed
        if compression_ratio < 0.3:
            # Need heavy compression
            scale = "scale=640:480"
            bitrate = "500k"
        elif compression_ratio < 0.6:
            # Medium compression
            scale = "scale=854:480"
            bitrate = "800k"
        else:
            # Light compression
            scale = "scale=1280:720"
            bitrate = "1200k"
        
        cmd = [
            'ffmpeg', '-i', input_path,
            '-vf', scale,
            '-b:v', bitrate,
            '-b:a', '128k',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-y',
            output_path
        ]
        
        print(f"Compressing video to ~{max_size_mb} MB...")
        subprocess.run(cmd, capture_output=True, check=True)
        
        # Check compressed size
        compressed_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Compressed video size: {compressed_size:.2f} MB")
        
        # Read and encode compressed video
        with open(output_path, 'rb') as f:
            video_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Clean up
        os.remove(output_path)
        
        return video_base64
        
    except Exception as e:
        print(f"Compression error: {str(e)}")
        if os.path.exists(output_path):
            os.remove(output_path)
        # Fall back to original
        with open(input_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html', endpoint_id=ENDPOINT_ID)

@app.route('/process', methods=['POST'])
def process_video():
    temp_file = None
    try:
        # Check if video file or URL
        video_url = request.form.get('video_url', '')
        video_base64 = None
        compression_used = False
        
        if 'video_file' in request.files:
            file = request.files['video_file']
            if file and allowed_file(file.filename):
                # Save temporarily
                temp_file = tempfile.mktemp(suffix=os.path.splitext(file.filename)[1])
                file.save(temp_file)
                
                # Get file size
                file_size_mb = os.path.getsize(temp_file) / (1024 * 1024)
                
                # Compress if needed (>10MB)
                if file_size_mb > 10:
                    print(f"Large file ({file_size_mb:.1f} MB), compressing...")
                    video_base64 = compress_video(temp_file, max_size_mb=8)
                    compression_used = True
                else:
                    # Small file, just encode
                    with open(temp_file, 'rb') as f:
                        video_base64 = base64.b64encode(f.read()).decode('utf-8')
                
                video_input = video_base64
                input_type = "file"
            else:
                return jsonify({'error': 'Invalid file type'}), 400
        elif video_url:
            video_input = video_url
            input_type = "url"
        else:
            return jsonify({'error': 'No video provided'}), 400
        
        # Get parameters
        prompt = request.form.get('prompt', 'Describe this video in detail')
        fps = float(request.form.get('fps', 1.0))
        max_frames = int(request.form.get('max_frames', 30))
        
        # Prepare RunPod request
        url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"
        headers = {"Authorization": f"Bearer {API_KEY}"}
        
        payload = {
            "input": {
                "video": video_input,
                "prompt": prompt,
                "fps": fps,
                "max_frames": max_frames
            }
        }
        
        # Calculate payload size
        payload_size_mb = len(json.dumps(payload)) / (1024 * 1024)
        print(f"Payload size: {payload_size_mb:.2f} MB")
        
        if payload_size_mb > 50:
            return jsonify({
                'success': False,
                'error': f'Video still too large after compression ({payload_size_mb:.1f} MB). Try reducing video length or quality.'
            }), 400
        
        # Send to RunPod
        print(f"Sending to RunPod - FPS: {fps}, Frames: {max_frames}, Compressed: {compression_used}")
        response = requests.post(url, json=payload, headers=headers, timeout=180)
        
        if response.status_code == 200:
            result = response.json()
            if "output" in result:
                response_data = {
                    'success': True,
                    'caption': result['output'].get('caption', 'No caption generated'),
                    'metadata': result['output'].get('metadata', {}),
                    'input_type': input_type,
                    'compression_used': compression_used
                }
                if compression_used:
                    response_data['metadata']['compressed'] = True
                return jsonify(response_data)
            else:
                return jsonify({
                    'success': False,
                    'error': 'Unexpected response format',
                    'response': result
                })
        else:
            return jsonify({
                'success': False,
                'error': f'RunPod error: {response.status_code}',
                'message': response.text[:500]
            }), response.status_code
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)

@app.route('/health')
def health():
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        ffmpeg_available = True
    except:
        ffmpeg_available = False
    
    return jsonify({
        'status': 'running',
        'endpoint_configured': bool(ENDPOINT_ID and API_KEY),
        'ffmpeg_available': ffmpeg_available,
        'max_upload_mb': 500
    })

if __name__ == '__main__':
    if not ENDPOINT_ID or not API_KEY:
        print("⚠️ Warning: RUNPOD_ENDPOINT_ID or RUNPOD_API_KEY not set in .env file")
    
    # Check ffmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✅ FFmpeg available for video compression")
    except:
        print("⚠️ FFmpeg not found - large files may fail")
        print("   Install with: brew install ffmpeg")
    
    print(f"🚀 Starting web interface on http://localhost:5000")
    print(f"📍 Using endpoint: {ENDPOINT_ID}")
    print(f"📦 Max file size: 500 MB (will compress if >10 MB)")
    app.run(debug=True, host='0.0.0.0', port=5000)