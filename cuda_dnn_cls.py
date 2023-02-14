import torch
import cv2
import os
import numpy as np

from torchvision import models

if __name__ == "__main__":

    torch_model = models.resnet50(pretrained=True)

    onnx_model_path = "models"
    onnx_model_name = "resnet50.onnx"
    
    os.makedirs(onnx_model_path,exist_ok=True)
    full_model_path = os.path.join(onnx_model_path, onnx_model_name)

    # generate model input
    generated_input = torch.randn(1, 3, 224, 224)

    # model export into ONNX format
    torch.onnx.export(
        torch_model,
        generated_input,
        full_model_path,
        verbose=True,
        input_names=["input"],
        output_names=["output"],
        opset_version=11
    )
    
    image_path = "test_images/squirrel_cls.jpg"

    #opencv API
    opencv_net = cv2.dnn.readNetFromONNX(full_model_path)
    opencv_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    opencv_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


    print(image_path)
    input_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    input_img = input_img.astype(np.float32)
    input_img = cv2.resize(input_img, (256, 256))
    mean = np.array([0.485, 0.456, 0.406]) * 255.0
    scale = 1 / 255.0
    std = [0.229, 0.224, 0.225]

    # prepare input blob to fit the model input:
    # 1. subtract mean
    # 2. scale to set pixel values from 0 to 1
    input_blob = cv2.dnn.blobFromImage(
        image=input_img,
        scalefactor=scale,
        size=(224, 224),  # img target size
        mean=mean,
        swapRB=True,  # BGR -> RGB
        crop=True  # center crop
    )  
    # 3. divide by std
    input_blob[0] /= np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
   
    opencv_net.setInput(input_blob)
    # opencv python api inference
    out = opencv_net.forward()
    print("OpenCV DNN prediction: \n")
    print("* shape: ", out.shape)
    imagenet_class_id = np.argmax(out)

    confidence = out[0][imagenet_class_id]
    print(imagenet_class_id,confidence)

