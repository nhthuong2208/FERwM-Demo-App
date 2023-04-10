from flask import Blueprint, render_template, request, jsonify, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import torch
from PIL import Image
from io import BytesIO
import re, base64
from torchvision import transforms
from utils import *
from urllib.parse import urlparse
import numpy as np
from configs import *
from networks import resnet18, CustomClassifier

main = Blueprint('main', __name__)
# {0: 'negative', 1: 'neutral', 2: 'positive'}


@main.route('/')
def index():
    return render_template('welcome.html')

@main.route('/hello')
def hello():
    pass

@main.route('/about')
def about():
    pass

@main.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    # `img` is reading the image from the given `img_path` using OpenCV's `cv2.imread()` function.
    filename = secure_filename(file.filename)
    if not os.path.exists(os.path.join(UPLOAD_PATH, 'uploads')):
        os.makedirs(UPLOAD_PATH + '/uploads')
    file.save(os.path.join(UPLOAD_PATH, 'uploads', filename))
    file_url = url_for('main.uploaded_file', filename=filename, _external=True)
    return jsonify(url=file_url)

@main.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(os.path.join(UPLOAD_PATH, 'uploads'), filename)

@main.route('/predict', methods=['POST'])
def predict_expression():
    if request.method == 'POST':
        image_str = request.json["image"]
        image_file = urlparse(image_str)
        image_path = os.path.join(UPLOAD_PATH, 'uploads', os.path.basename(image_file.path))


        with open(image_path, "rb") as fh:
            image_data = BytesIO(fh.read())
        img = Image.open(image_data)
        img.save("sample" + '.jpg', "JPEG")
        
        # Get response
        response = get_info("sample" + '.jpg', "enet", "enet")
        return jsonify(response)

@main.route('/predict-video', methods=['POST'])
def predict_expression_video():
    if request.method == 'POST':
        image_str = request.json["image"]

        base64_data = re.sub('^data:image/jpeg;base64,', '', image_str)
        byte_data = base64.b64decode(base64_data)
        image_data = BytesIO(byte_data)
        img = Image.open(image_data)
        img.save("sample" + '.jpg', "JPEG")
        
        # Get response
        response = get_info("sample" + '.jpg', "enet", "enet")
        return jsonify(response)
