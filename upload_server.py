#!/usr/bin/env python3
"""
Simple video upload server that hosts videos locally
This avoids sending large files to RunPod
"""

from flask import Flask, request, jsonify, send_file
import os
import tempfile
import uuid
from werkzeug.utils import secure_filename
import threading
import time

app = Flask(__name__)

# Store uploaded videos temporarily
VIDEO_STORAGE = {}
CLEANUP_AFTER = 300  # 5 minutes

app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

def cleanup_old_videos():
    """Remove videos older than 5 minutes"""
    while True:
        current_time = time.time()
        to_remove = []
        for video_id, data in VIDEO_STORAGE.items():
            if current_time - data['timestamp'] > CLEANUP_AFTER:
                if os.path.exists(data['path']):
                    os.remove(data['path'])
                to_remove.append(video_id)
        for video_id in to_remove:
            del VIDEO_STORAGE[video_id]
        time.sleep(60)  # Check every minute

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_old_videos, daemon=True)
cleanup_thread.start()

@app.route('/upload', methods=['POST'])
def upload():
    """Upload video and get a URL"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Generate unique ID
    video_id = str(uuid.uuid4())
    
    # Save file
    temp_path = tempfile.mktemp(suffix=os.path.splitext(file.filename)[1])
    file.save(temp_path)
    
    # Store reference
    VIDEO_STORAGE[video_id] = {
        'path': temp_path,
        'filename': secure_filename(file.filename),
        'timestamp': time.time()
    }
    
    # Return URL
    video_url = f"http://localhost:5001/video/{video_id}"
    
    return jsonify({
        'success': True,
        'video_id': video_id,
        'video_url': video_url,
        'message': 'Video uploaded. URL valid for 5 minutes.'
    })

@app.route('/video/<video_id>')
def serve_video(video_id):
    """Serve uploaded video"""
    if video_id not in VIDEO_STORAGE:
        return jsonify({'error': 'Video not found or expired'}), 404
    
    video_data = VIDEO_STORAGE[video_id]
    return send_file(video_data['path'], mimetype='video/mp4')

if __name__ == '__main__':
    print("🎬 Video Upload Server")
    print("This server hosts videos locally to avoid large uploads to RunPod")
    print("Videos are automatically deleted after 5 minutes")
    print("-" * 50)
    print("Running on: http://localhost:5001")
    app.run(port=5001, debug=False)