from fastapi import FastAPI, File, Form, UploadFile
from typing import List
from torchvision import transforms as T
import torch 
import PIL.Image as Image
import io
from torchvision import models
import logging
import time

#lets think sequentially from to import images to process images to return images

app = FastAPI()

logging.basicConfig(level=logging.INFO,
     format='%(asctime)s:%(levelname)s:%(message)s')

def process_single_image(file: UploadFile):
    content = file.file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    transform_test = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
    ])
    Transform_inference = transform_test(image)
    #bring model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = models.resnet18(pretrained=False)
    num_of_classes = 200
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_of_classes)
    PATH = torch.load('resnet18_finetuned_imagenette.pth', map_location=torch.device(device))
    new_state_dict = {}
    for key , value in PATH.items():
        if key.startswith('module.'):
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value
   
    model.load_state_dict(new_state_dict, strict=False)
  
    
    model.eval()
    model = model.to(device)
    start_time = time.time()
    Transform_inference = Transform_inference.to(device)
    with torch.no_grad():
       predictions = model(Transform_inference.unsqueeze(0))
    _, predicted = torch.max(predictions, 1)
    #wite inference speed to a log file
    END_time = time.time()
    inference_speed = END_time - start_time
    logging.info(f"INference speed:{inference_speed:.4f}seconds")

    return predicted.item()

# to upload images (multiple)
@app.post("/input/")
async def read_input(files : List[UploadFile] = File(...)):
    result = []

    for file in files:
        prediction = process_single_image(file)
        #map prediction to class name
        with open('class_names_mapping.txt', 'r') as f:
            class_mapping = {}
            for line in f:
                idx, class_name = line.strip().split('\t')
                class_mapping[int(idx)] = class_name
        prediction = class_mapping.get(prediction, "Unknown")
        result.append(prediction)


        await file.close()
    
    return {"message": "Files processed successfully", "predictions": result}
    #here we will write code to accept images from user ,may be single image or folder of images
     
