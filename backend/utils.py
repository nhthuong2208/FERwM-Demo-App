import torch
import os
from configs import *
from networks import resnet18, CustomClassifier
from PIL import Image
from io import BytesIO
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize
import torch.nn.functional as F
import base64
import torch
import torch.nn as nn



def get_model_layer(model_id: str = "dacl"):
    model, layer = None, None
    if model_id == 'enet':
        model = torch.load(os.path.join(MODEL_PATH, "enet_mtl.pt"), map_location=device)['model']
        layer = model.bn2, # last layer

    elif model_id == 'fan':
        model = torch.load(os.path.join(MODEL_PATH, "fan_ms1m.pt"), map_location=device)['model']
        layer = model.layer4[-1], # last layer
    elif model_id == 'dacl':
        model = resnet18(pretrained=os.path.join(MODEL_PATH, 'dacl.pth'))
        layer = None # can not use gradcam

    return model, layer[0]


def get_gradcam(img_path: str, model_id: str = "enet", method_id: str = 'gradcam++') -> None:
    """Return the image
    
    """
    model, layer = get_model_layer(model_id)
    model.eval() # switch model to eval mode
    # print(layer)
    if method_id not in METHOD_GRADCAM.keys():
        raise ValueError("Method id is not in METHOD_GRADCAM dictionary")
    
    cam = METHOD_GRADCAM[method_id](model=model, target_layer=layer, use_cuda=False) # load grad-cam
    img = cv2.imread(img_path, 1) # read image ???
    
    resized = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA) # resize image to (224,224)

    rgb_img = resized[:, :, ::-1] # BGR to RGB conversion
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225]) # preprocessing input
    
    grayscale_cam = cam(input_tensor=input_tensor,target_category=None,aug_smooth=False,eigen_smooth=False)

    grayscale_cam = grayscale_cam[0, :]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR) #BGR image

    target_path = f"{img_path.split('.')[0]}_gradcam.jpg"
    cv2.imwrite(target_path, cam_image)




def get_predict(model, path: str) -> dict:
    response = dict()
    model.eval()
    # try:
    with torch.no_grad():
        # img_path = PATH + "sample" + '.jpg'
        img_path = path
        img = Image.open(img_path)


        # Resize and convert to tensor
        transform = transforms.Compose([Resize((224, 224)), ToTensor()])

        # convert = transforms.ToTensor()
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        output = model(img_tensor) # ?

        print(torch.argmax(output, dim=1).item())
        response['predict'] = EMOTION_INDEX[torch.argmax(output, dim=1).item()]
        
        probs = F.softmax(output, dim=1)
        print("Probs:", probs)
        probs = probs.cpu().numpy()
        response['probs'] = {"negative" : f'{probs[0][0]}',
                            "neutral" : f'{probs[0][1]}',
                            "positive" : f'{probs[0][2]}'}
        
    
    return response





def get_info(img_path: str = "samples/happy-1.png", model_predict_id: str = 'enet', model_gradcam_id: str = 'enet'):
    model, _ = get_model_layer(model_predict_id)
    model.eval()

    response = get_predict(model, img_path)

    # For gradcam
    get_gradcam(img_path, model_gradcam_id)

    gradcam_path = f"{img_path.split('.')[0]}_gradcam.jpg"
    with open(gradcam_path, 'rb') as f:
        image = f.read()

    encoded_image = base64.b64encode(image).decode('utf-8')
    response['gradcam'] = encoded_image
    print(response)
    return response


# get_info("samples/happy-1.png")