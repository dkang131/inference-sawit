from flask import Flask, render_template
from ultralytics import YOLO
from datetime import datetime
import pytz
import logging
import cv2
import torch
import os

from extensions import db
from models import Video

app = Flask(__name__)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def get_available_models():
    """Get list of available models from the model directory"""
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')
    models = []
    
    if os.path.exists(model_dir):
        for item in os.listdir(model_dir):
            item_path = os.path.join(model_dir, item)
            if os.path.isfile(item_path) and (item.endswith('.pt') or item.endswith('.pth') or item.endswith('.onnx')):
                models.append(item)
    
    # If no models found, add a default option
    if not models:
        models = ['No models available']
    
    return models

@app.route('/')
def index():
    models = get_available_models()
    return render_template('index.html', models=models)

if __name__ == '__main__':
    app.run(debug=True, port=4786, host='0.0.0.0')