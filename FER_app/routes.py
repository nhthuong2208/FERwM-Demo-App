from flask import Blueprint, render_template, request, jsonify, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import torch
from PIL import Image
from io import BytesIO
import re, base64
from torchvision import transforms
from .utils import *
from urllib.parse import urlparse
import numpy as np

main = Blueprint('main', __name__)

PATH = os.path.join(os.getcwd(), 'FER_app/data')
UPLOAD_PATH = os.path.join(os.getcwd(), 'FER_app/static/images')

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
        model_chosen = get_selected_model(PATH, "Multi-task EfficientNet-B2")
        model_chosen.eval()

        # base64_data = re.sub('^data:image/jpeg;base64,', '', image_str)
        # byte_data = base64.b64decode(base64_data)
        # image_data = BytesIO(byte_data)
        with open(image_path, "rb") as fh:
            image_data = BytesIO(fh.read())
        img = Image.open(image_data)
        img.save(PATH + "sample" + '.jpg', "JPEG")
        convert = transforms.ToTensor()

        with torch.no_grad():
            predict = model_chosen(convert(Image.open(PATH + "sample" + '.jpg')).unsqueeze(0).to(torch.device('cpu')))
        if isinstance(predict, torch.Tensor):
            result = predict.cpu().numpy()
            response = {"negative":f'{result[0][0]}',"neutral":f'{result[0][1]}',"positive":f'{result[0][2]}'}
            return jsonify(response)
        return {"success":"false"}

@main.route('/predict-video', methods=['POST'])
def predict_expression_video():
    if request.method == 'POST':
        image_str = request.json["image"]

        model_chosen = get_selected_model(PATH, "Multi-task EfficientNet-B2")
        model_chosen.eval()

        base64_data = re.sub('^data:image/jpeg;base64,', '', image_str)
        byte_data = base64.b64decode(base64_data)
        image_data = BytesIO(byte_data)
        img = Image.open(image_data)
        img.save(PATH + "sample" + '.jpg', "JPEG")
        convert = transforms.ToTensor()

        with torch.no_grad():
            predict = model_chosen(convert(Image.open(PATH + "sample" + '.jpg')).unsqueeze(0).to(torch.device('cpu')))
        if isinstance(predict, torch.Tensor):
            result = predict.cpu().numpy()
            response = {"negative":f'{result[0][0]}',"neutral":f'{result[0][1]}',"positive":f'{result[0][2]}'}
            return jsonify(response)
        return {"success":"false"}
