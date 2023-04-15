import torch
import os
from configs import *
from networks import *
from PIL import Image
from io import BytesIO
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize
import torch.nn.functional as F
import base64
import torch
import torch.nn as nn
import math


def get_model_layer(model_id: str = "dacl"):
    """Get model and layer using model id

    We load whole model using model architecture and pretrained file for parameters
    Now we support 3 following model id for this api:
        + 'enet': the model with EfficientNetB2 backbone is trained with Multitask learing and trained with FERwM task which wrapped as a file at path models/enet_mtl.pt
            > More information found on: https://github.com/HSE-asavchenko/face-emotion-recognition
        + 'fan_ms1m': the model with FAN (Frame Attention Network) architecture is trained with our team' s data for FERwM task is saved at path models/fan_ms1m.pt
            > More information found on: https://github.com/Open-Debin/Emotion-FAN
        + 'dacl': the model with Resnet18 customization with attention idea and train with task FERwM is saved at path models/dacl.pth
            > More information found on: https://github.com/amirhfarzaneh/dacl
    
    With each model, we want to retrieve the (final) layer which is used for visualizing the GradCam (the heat map that points CNN focus on which part of image)
        > More information about GradCam lib: https://github.com/jacobgil/pytorch-grad-cam
    We just support 2 following model id for retrieving the last layer:
        + 'enet': layer bn2
        + 'fan': last layer inside group layer4
    

    Args:
        model_id (str): A string value identifying the model. Only supported: enet, fan, dacl
    
    Returns:
        model: The pytorch model with fully params load from pretrained file
        layer: The layer of model used for GradCam
        
    """
    model, layer = None, None
    if model_id == 'enet':
        model = torch.load("models/enet_mtl.pt", map_location=device)['model']
        layer = model.bn2, # last layer
    elif model_id == 'fan':
        model = torch.load(os.path.join(MODEL_PATH, "fan_ms1m.pt"), map_location=device)['model']
        layer = model.layer4[-1], # last layer
    elif model_id == 'dacl':
        model = resnet18(pretrained=os.path.join(MODEL_PATH, 'dacl.pth'))
        layer = [None] # can not use gradcam

    return model, layer[0]


def get_gradcam(img: np.ndarray, model_id: str = "enet", method_id: str = 'gradcam++', 
                save_gradcam: bool = False) -> str:
    """Get gradcam result from an image

    Switch the mode of model into eval state. Resize image to (224, 224) and do some simple preprocessing. Use show_cam_on_image of gradcam-pytorch lib.
    The gradcam image after transform is returned and depends on save_gradcam logic variable for save or not that image to local.

    Args:
        img (np.ndarray with shape (H,W,3)): The image after loaded using cv2
        model_id (str): The model that used for gradcam computation. We suport only 2 values: enet and fan. The default value is enet
        method_id (str): The method that used for gradcam computation. We support some methods including: gradcam, scorecam, gradcam++, ablationcam, xgradcam, eigencam, eigengradcam. The default value is gradcam++.
        save_gradcam (bool): Whether save the gradcam image to local or not
        
    Returns:
        cam_str (str): The string representation of the gradcam image which is encoded to base64 format
        
    """
    model, layer = get_model_layer(model_id)
    model.eval() # switch model to eval mode

    if method_id not in METHOD_GRADCAM.keys():
        raise ValueError("Method id is not in METHOD_GRADCAM dictionary")
    cam = METHOD_GRADCAM[method_id](model=model, target_layer=layer, use_cuda=False) # load grad-cam

    
    resized = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA) # resize image to (224,224)
    rgb_img = resized[:, :, ::-1] # BGR to RGB conversion
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225]) # preprocessing input
    
    grayscale_cam = cam(input_tensor=input_tensor,target_category=None,aug_smooth=False,eigen_smooth=False)

    grayscale_cam = grayscale_cam[0, :]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR) #BGR image

    if save_gradcam:
        gradcam_path = f"{img_path.split('.')[0]}_gradcam.jpg"
        cv2.imwrite(gradcam_path, cam_image)
    
    # Convert to decode base64
    cam_str = base64.b64encode(cv2.imencode('.jpg', cam_image)[1]).decode('utf-8')
    return cam_str



def get_predict(model, img: np.ndarray, response: dict) -> None:
    """Get predictions from given model and given image

    Initially, we have the model which have been loaded and an image shape (H,W,3) - Height, Width, #Chanel (R-G-B).
    At first, we switch the mode of the model into eval state and torch.no_grad mode (for calculate the output without change anything). So we change the shape of image from numpy ndarray into tensor with the same shape using transform of torchvision.
    Next, we pass this tensor into model and get the output tensor. Now the output is messy and result of the last fully connected layer.
    Finally, we apply softmax function from pytorch onto output tensor and get the final probability of 3 emotions. And use argmax function for this tensor to get emotion of this prediction.
    > Note: The index of tensor match which each emotion is {0: 'negative', 1: 'neutral', 2: 'positive'}

    Args:
        model: The model that have been loaded
        img (np.ndarray with shape (H,W,3)): The image after loaded using cv2
        response (dict): The dictionary containing the current value as reference and we want to update this
    
    Returns: None
        But we need to update the result into response dictionary with Python dictionary format
        {
            'predict': $which emotion that model predict$
            'probs': {
                'negative': $the probability for negative emotion$
                'neutral': $the probability for neutral emotion$
                'positive': $the probability for positive emotion$
            }
        }

    """

    model.eval() # switch model to eval mode

    with torch.no_grad():
        # Resize and convert to tensor
        transform = transforms.Compose([ToTensor()])
        img = cv2.resize(img, (224, 224))
        img_tensor = transform(img).unsqueeze(0).to(device) 
        output = model(img_tensor)

        response['predict'] = EMOTION_INDEX[torch.argmax(output, dim=1).item()]
        probs = F.softmax(output, dim=1)
        probs = probs.cpu().numpy()
        response['probs'] = {"negative" : f'{probs[0][0]}',
                            "neutral" : f'{probs[0][1]}',
                            "positive" : f'{probs[0][2]}'}
        






def preprocess(img: np.ndarray, save_crop: bool = False, save_align: bool = False,
               encode_crop: bool = False, encode_align: bool = False) -> tuple[np.ndarray, dict]:
    """Preprocessing the given image
    
    The process is split into 2 parts: crop and align
        1/ Crop: Use dlib library and get_frontal_face_detector function for detect all the faces appear in the image
            > Catch error with number of face detected
                - =0: No faces detected
                - >=2 : Too many faces detected
                - =1: Next step
        2/ Align: May be the face after crop is not straightforward. We need to rotate it base on the the baseline - the line between the left eye and right eye which is the point 36 and point 45 in 68-point landmark. We calculate the angle between the baseline and horizontal axis and then rotate the image with this angle
            > Catch error if the part 36 and 45 is None (not detected)
    Args:
        img (np.ndarray with shape (H,W,3)): The image after loaded using cv2
        save_crop (bool): Whether to save the cropped image to local. The default is False
        save_align (bool): Whether to save the alignment image to local. The default is False
        encode_crop (bool): Whether to save the encode base64 of the cropped image in the response. The default is False
        encode_align (bool): Whether to save the encode base64 of the alginment image in the response. The default is False
    
    Returns: tuple[np.ndarray, dict]
        align_img (np.ndarray with shape (H,W,3)): The image after preprocessing
        response (dict): The dictionary of the response. This follows the format
            {
                'message': $the message$
                'crop' : $base64 img or None$
                'align' : $base64 img or None$
            }
        
    """


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    response = dict()


    # Detect the face in the image using dlib.get_frontal_face_detector()
    faces = detector(gray)

    # Raise exception if the number of faces is not equal to 1
    if len(faces) == 0:
        return (img, {'message': 'No faces detected'})
    if len(faces) >= 2:
        return (img, {'message': 'Too many faces detected'})

    # Get the only face and its boundary
    face = faces[0]
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()

    # crop the face image
    crop_img = img[y1:y2, x1:x2]

    # Save the crop image
    if save_crop:
        crop_path = f"{img_path.split('.')[0]}_crop.jpg"
        cv2.imwrite(crop_path, crop_img)

    if encode_crop:
        crop_str = base64.b64encode(cv2.imencode('.jpg', crop_img)[1]).decode('utf-8')
        response['crop'] = crop_str


    # Get the facial landmarks
    landmarks = predictor(gray, face)
    if landmarks.part(36) is None:
        return (img, {'message': 'Landmarks point 36 is not detected'})
    if landmarks.part(45) is None:
        return (img, {'message': 'Landmarks point 45 is not detected'})
    
    # Get the coordinates of the center of the two eyes
    x1, y1 = landmarks.part(36).x, landmarks.part(36).y
    x2, y2 = landmarks.part(45).x, landmarks.part(45).y
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    
    # Calculate the angle between the line connecting the center of the two eyes and a horizontal line
    angle = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi
    
    # Rotate the crop image by the calculated angle
    rows, cols = crop_img.shape[:2]
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1)
    align_img = cv2.warpAffine(crop_img, M, (cols, rows))
    
    # Save the align image
    if save_align:
        align_path = f"{img_path.split('.')[0]}_align.jpg"
        cv2.imwrite(align_path, align_img)
    
    if encode_align:
        align_str = base64.b64encode(cv2.imencode('.jpg', align_img)[1]).decode('utf-8')
        response['align'] = align_str


    # Return final image for next prediction
    response['message'] = 'Process image successfully!!'
    return (align_img, response)




def get_info(_img_path: str = "samples/happy-1.png", model_predict_id: str = 'dacl', 
             model_gradcam_id: str = 'enet', method_gradcam_id: str = 'gradcam++',
             save_crop: bool = False, save_align: bool = False, save_gradcam: bool = False,
             encode_crop: bool = False, encode_align: bool = False, encode_gradcam: bool = True) -> dict:
    """Final function of APIs calling processing a image from a link
    
    Gathering all the information from other function to process a image and get the prediction
        > From the image path, Load the image using cv2. 
        > From model_predict_id, Load model using get_model_layer() function
        > From model_gradcam_id, method_gradcam_id, save_gradcam and encode_gradcam, Transform the image into gradcam image and save image to local of return as a part of response
        > In the preprocessing step, save_crop, save_align, encode_crop and encode_align parameters for save image to local or encode and returned it as a part of response for each progress: crop and align


    Args:
        _img_path (str): The local path image for processing. The default is samples/happy-1.png for testing. Usually data/sample.jpg
        model_predict_id (str): The model used for prediction. The default is dacl
        model_gradcam_id (str): The model used for gradcam calculation. The default is enet
        method_gradcam_id (str): The method used for gradcam calculation. The default is gradcam++
        save_crop (bool): Whether to save the cropped image to local. The default is False
        save_align (bool): Whether to save the alignment image to local. The default is False
        save_gradcam (bool): Whether to save the gradcam image to local. The default is False
        encode_crop (bool): Whether to save the encode base64 of the cropped image in the response. The default is False
        encode_align (bool): Whether to save the encode base64 of the algin image in the response. The default is False
        encode_gradcam (bool): Whether to save the encode base64 of the gradcam image in the response. The default is True

    Returns: dict
        response (dict): The final result returned from processing. The format of this dictionary follows:
            {
                'message': $the message$
                'crop' : $base64 img or None$
                'align' : $base64 img or None$
                'predict': $which emotion that model predict$
                'probs': {
                    'negative': $the probability for negative emotion$
                    'neutral': $the probability for neutral emotion$
                    'positive': $the probability for positive emotion$
                }
            }
    
    """


    global img_path, model
    img_path = _img_path
    img = cv2.imread(img_path)
    # print(img.shape)
    (img, response) = preprocess(img, save_crop, save_align, encode_crop, encode_align) #True, True

    if model is None:
        model, _ = get_model_layer(model_predict_id)
        model.eval()

    get_predict(model, img, response)

    # For gradcam
    if encode_gradcam:
        can_str = get_gradcam(img, model_gradcam_id, method_gradcam_id, save_gradcam)
        response['gradcam'] = can_str

    return response


def benchmark():
    import time
    start = time.time()

    # print(get_info("samples/happy-1.png", "dacl", "enet", 'gradcam++', True, True, True, True, True))
    get_info("samples/happy-1.png")

    end = time.time()
    print('Total time is:', end - start)

