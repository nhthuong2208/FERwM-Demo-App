"""
[NOTE] Using command: "pip install grad-cam==1.3.1"
to install the pytorch_grad_cam package
"""

import os
import glob
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