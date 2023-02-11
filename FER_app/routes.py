from flask import Blueprint, render_template, request, jsonify
import os
import torch
from PIL import Image
from io import BytesIO
import re, base64
from torchvision import transforms
from .utils import *

main = Blueprint('main', __name__)

PATH = os.path.join(os.getcwd(), 'FER_app/data')

@main.route('/')
def index():
    return render_template('welcome.html')

@main.route('/hello')
def hello():
    pass

@main.route('/about')
def about():
    pass

@main.route('/predict', methods=['POST'])
def predict_expression():
    if request.method == 'POST':
        data = request.get_json()
        image_str = data["image"]

        model_chosen = get_selected_model(PATH, data["model"])
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
