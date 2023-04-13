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
                save_gradcam: bool = False, encode_gradcam: bool = True) -> None:
    """Return the image
    
    """
    model, layer = get_model_layer(model_id)
    model.eval() # switch model to eval mode
    # print(layer)
    # print(method_id)
    # print(METHOD_GRADCAM.keys())
    if method_id not in METHOD_GRADCAM.keys():
        raise ValueError("Method id is not in METHOD_GRADCAM dictionary")
    
    cam = METHOD_GRADCAM[method_id](model=model, target_layer=layer, use_cuda=False) # load grad-cam
    # img = cv2.imread(img_path, 1) # read image ???
    # img shape(x,x,3)
    
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
    if encode_gradcam:
        cam_str = base64.b64encode(cv2.imencode('.jpg', cam_image)[1]).decode('utf-8')
    return cam_str



def get_predict(model, img: np.ndarray, response: dict) -> dict:
    # response = dict()
    model.eval()

    with torch.no_grad():
        # Resize and convert to tensor
        transform = transforms.Compose([ToTensor()])

        img = cv2.resize(img, (224, 224))

        # convert = transforms.ToTensor()
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        output = model(img_tensor) # ?

        # print(torch.argmax(output, dim=1).item())
        response['predict'] = EMOTION_INDEX[torch.argmax(output, dim=1).item()]
        
        probs = F.softmax(output, dim=1)
        # print("Probs:", probs)
        probs = probs.cpu().numpy()
        response['probs'] = {"negative" : f'{probs[0][0]}',
                            "neutral" : f'{probs[0][1]}',
                            "positive" : f'{probs[0][2]}'}
        






# Final function
def preprocess(img: np.ndarray, save_crop: bool = False, save_align: bool = False,
               encode_crop: bool = False, encode_align: bool = False):
    # img_path: str = 'samples/happy-1.png'
    # Read the image using cv2
    # img = cv2.imread(img_path)
    # img.shape = (304,310,3)
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
             encode_crop: bool = False, encode_align: bool = False, encode_gradcam: bool = True):
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
    # print()
    get_info("samples/happy-1.png")

    end = time.time()
    print('Total time is:', end - start)

benchmark()

# res = get_info("samples/happy-1.png")
# print(res)