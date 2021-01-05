import io
import json
import math

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from flask import Flask, jsonify, request
from flask_cors import CORS


class_index = json.load(open('class_index.json'))
model = torch.load("mixmodel97(full).pth", map_location=torch.device("cpu"))
model.eval()


def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize(128),          
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def truncate(f, n):
    return math.floor(f * 10 ** n) / 10 ** n

def get_prediction(image_bytes):
    tensor_img = transform_image(image_bytes=image_bytes)
    outputs = F.softmax(model(tensor_img), 1)
    pred = torch.argmax(outputs, 1)
    predicted_idx = str(pred.item())

    values = outputs.topk(5).values.detach().numpy()[0]
    indices = outputs.topk(5).indices.detach().numpy()[0]
    top5 = {class_index[str(i)]:truncate(j, 5) * 100 for i, j in zip(indices, values)}

    return class_index[predicted_idx], values


app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_name, top_class = get_prediction(image_bytes=img_bytes)
        
        return jsonify({'class_name': class_name})

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'hello':'test success'})


if __name__ == '__main__':
    app.run()
