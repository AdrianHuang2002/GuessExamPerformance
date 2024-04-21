from __future__ import division
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import cv2

import argparse
import random
from pathlib import Path

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import PIL.Image

import util.misc as utils
from models import build_model

from main import get_args_parser

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import io
from PIL import Image
import torch

from flask import Flask, redirect, render_template, request, url_for, send_from_directory


fdModel = torch.load('detr_model_full.pth')
fdModel.eval()

app = Flask(__name__)

app.config['FILE_LIMIT'] = 16 * 1024 * 1024

ALLOWED_FILES = set(['mp4'])

json_file = open('fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
erModel = model_from_json(loaded_model_json)
erModel.load_weights("fer.h5")

WIDTH = 48
HEIGHT = 48
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_FILES

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def detect(im, model, transform):
  img = transform(im).unsqueeze(0)
  assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'Demo model only supports images up to 1600 on each side.'
  outputs = model(img)
  probas = outputs['pred_logits'].softmax(-1)[0,:,:-1]
  keep = probas.max(-1).values > 0.7
  bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0,keep], (800, 400))
  return probas[keep], bboxes_scaled

def process_single_frame(frame):
    scores, boxes = detect(frame, fdModel, transform)
    gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)

    emotion = []

    for (x, y, xmax, ymax) in boxes:
        if x < 0 or y < 0  or ymax < 0 or xmax < 0:
            continue
        roi_gray = gray[int(y):int(ymax), int(x):int(xmax)]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)

        #predicting the emotion
        yhat= erModel.predict(cropped_img)
        emotion.append(labels[int(np.argmax(yhat))])
    return (frame, boxes.tolist(), emotion)

def label_emotion(pil_img, b_boxes, emotion, is_ground_truth=False):
    plt.figure(figsize=(16, 10))
    plt.axis('off')
    ax = plt.gca()
    ax.imshow(pil_img)

    for (xmin, ymin, xmax, ymax), e in zip(b_boxes, emotion):
        xmax = xmax if is_ground_truth else xmax - xmin
        ymax = ymax if is_ground_truth else ymax - ymin
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax, ymax, fill=False, edgecolor='red', linewidth=3))
        ax.text(xmin, ymin, e, fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5))

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img = Image.open(buf)
    # Do not close buf here if img needs to be used outside this function
    return img


def process_video(in_path, out_path):
    counts = {"Angry": 0, "Disgust":0, "Fear":0, "Happy":0, "Sad":0, "Surprise":0, "Neutral":0}
    print("Processing your video")

    cap = cv2.VideoCapture(in_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID' or 'MJPG'
    out = cv2.VideoWriter(out_path, fourcc, fps, (800, 400))  # Adjust the size to match resized frames

    step = 20
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (800, 400))
        b = []
        e = []
        if count % step == 0:
            f, b, e = process_single_frame(frame)
            counts[e[0]] += 1
            pil_image = label_emotion(frame, b, e)
            processed_frame = np.array(pil_image)
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            frame = processed_frame
        
        
        # Write the processed frame to the output video
        if frame.shape[1] != 800 or frame.shape[0] != 400:
            frame = cv2.resize(frame, (800, 400))
        out.write(frame)
        count += 1
        
    # Release everything if job is finished
    cap.release()
    out.release()
    print("Output video done")

    vals = counts.values()
    total = 0
    for v in vals:
        total += v
    plus = counts['Happy'] + counts['Neutral'] + counts['Surprise']

    percent = float(plus)/total

    result = ""
    if percent < 0.001:
        result = "You'd better think of a good explanation to your parents on this exam"
    elif percent < 0.002:
        result = "It's ok you will do better next time"
    elif percent < 0.003:
        result = "Why are you not getting the A"
    else:
        result = "Good job"


    return result


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return redirect(request.url)
    
    file = request.files['video']
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        file.save('static/videos/' + file.filename)

        path = 'static/videos/' + file.filename
        
        processed_name = "processed_" + file.filename
        processed_path = 'static/processed/' + processed_name
        result = process_video(path, processed_path)
        
        return render_template('index.html', processed=processed_name, filename=processed_path, result=result)
    return "invalid file"
        

@app.route('/display_video/<filename>')
def display_video(filename):
    return send_from_directory('static/processed', filename)

if __name__ == "__main__":
    app.run()
