#!/usr/bin/env python3
"""
Flask web interface for video-SALMONN 2 RunPod testing
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import base64
import requests
import json
from dotenv import load_dotenv
import tempfile
from werkzeug.utils import secure_filename

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
ENDPOINT_ID = os.getenv('RUNPOD_ENDPOINT_ID')
API_KEY = os.getenv('RUNPOD_API_KEY')
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'webm', 'mkv'}

app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html', endpoint_id=ENDPOINT_ID)

@app.route('/process', methods=['POST'])
def process_video():
    try:
        # Check if video file or URL
        video_url = request.form.get('video_url', '')
        video_base64 = None
        
        if 'video_file' in request.files:
            file = request.files['video_file']
            if file and allowed_file(file.filename):
                # Read and encode file
                video_base64 = base64.b64encode(file.read()).decode('utf-8')
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
        
        # Send to RunPod
        print(f"Sending request to RunPod - FPS: {fps}, Frames: {max_frames}")
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            if "output" in result:
                # Handle double-wrapped output
                output_data = result['output']
                if isinstance(output_data, dict) and 'output' in output_data:
                    output_data = output_data['output']
                
                return jsonify({
                    'success': True,
                    'caption': output_data.get('caption', 'No caption generated'),
                    'metadata': output_data.get('metadata', {}),
                    'input_type': input_type
                })
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
                'message': response.text
            }), response.status_code
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'running',
        'endpoint_configured': bool(ENDPOINT_ID and API_KEY)
    })

if __name__ == '__main__':
    if not ENDPOINT_ID or not API_KEY:
        print("⚠️ Warning: RUNPOD_ENDPOINT_ID or RUNPOD_API_KEY not set in .env file")
    
    print(f"🚀 Starting web interface on http://localhost:5000")
    print(f"📍 Using endpoint: {ENDPOINT_ID}")
    app.run(debug=True, host='0.0.0.0', port=5000)