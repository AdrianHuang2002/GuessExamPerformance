from flask import Flask, flash, request, redirect, url_for, render_template
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from keras.models import model_from_json

app = Flask(__name__)

FOLDER = 'static/uploads/'
app.secret_key = "secret key"
app.config['FOLDER'] = FOLDER
app.config['FILE_LIMIT'] = 16 * 1024 * 1024

ALLOWED_FILES = set(['png', 'jpg', 'jpeg', 'gif'])

json_file = open('fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("fer.h5")
print("Loaded model from disk")

WIDTH = 48
HEIGHT = 48
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_FILES

def process_image(file_path):
    full_size_image = cv2.imread(file_path)
    print("Image Loaded")
    gray = cv2.cvtColor(full_size_image, cv2.COLOR_RGB2GRAY)
    face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face.detectMultiScale(gray, 1.3, 10)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (WIDTH, HEIGHT)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        cv2.rectangle(full_size_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        yhat = loaded_model.predict(cropped_img)
        cv2.putText(full_size_image, labels[int(np.argmax(yhat))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        print("Emotion: " + labels[int(np.argmax(yhat))])

    return full_size_image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('Empty selection')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['FOLDER'], filename)
        file.save(file_path)
        
        processed_image = process_image(file_path)
        
        flash('Displaying uploaded image:')
        return render_template('index.html', filename=filename, processed_image=processed_image)
    else:
        flash('Only png, jpg, jpeg, gif files are allowed')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename="uploads/" + filename), code=301)

if __name__ == "__main__":
    app.run()
