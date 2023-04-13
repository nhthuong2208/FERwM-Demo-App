"""
[NOTE] Using command: "pip install grad-cam==1.3.1"
to install the pytorch_grad_cam package
"""

import os
import glob
import dlib
import cv2
import torch
import numpy as np
from networks import resnet18, CustomClassifier

from pytorch_grad_cam import GradCAM, \
                             ScoreCAM, \
                             GradCAMPlusPlus, \
                             AblationCAM, \
                             XGradCAM, \
                             EigenCAM, \
                             EigenGradCAM

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

METHOD_GRADCAM = {"gradcam": GradCAM,
                    "scorecam": ScoreCAM,
                    "gradcam++": GradCAMPlusPlus,
                    "ablationcam": AblationCAM,
                    "xgradcam": XGradCAM,
                    "eigencam": EigenCAM,
                    "eigengradcam": EigenGradCAM}

EMOTION_INDEX = {0: 'negative', 1: 'neutral', 2: 'positive'}

PATH = os.path.join(os.getcwd(), 'data')
UPLOAD_PATH = os.path.join(os.getcwd(), 'FER_app/static/images')
MODEL_PATH = os.path.join(os.getcwd(), 'models')

# Load the face detection model
# face_cascade = cv2.CascadeClassifier('dlib_models/haarcascade_frontalface_default.xml')

# Load the facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("dlib_models/shape_predictor_68_face_landmarks.dat")

img_path = ''
model = None