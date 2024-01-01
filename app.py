# app.py
from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

app = Flask(__name__)
model = tf.keras.saving.load_model('heart-disease-predictor.h5')
def crop_image(image_path, left=71.5, top=287.5, right=2102, bottom=1228):
    cropped_img = Image.open(image_path).crop((left, top, right, bottom))
    return cropped_img

def bg_remov(image_array):
    num_arr = np.array(image_array)
    if len(num_arr.shape) == 2:  # Grayscale image
        result = salt(image_array, 10)
        median = cv2.medianBlur(result, 5)
        (thresh, blackAndWhiteImage) = cv2.threshold(median, 85, 255, cv2.THRESH_BINARY)
    elif len(num_arr.shape) == 3 and image_array.shape[2] == 3:  # RGB image
        result = salt(image_array, 10)
        median = cv2.medianBlur(result, 5)
        gray = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
        (thresh, blackAndWhiteImage) = cv2.threshold(gray, 85, 255, cv2.THRESH_BINARY)
    else:
        raise ValueError("Unsupported image format. Expected grayscale (2D) or RGB (3D) image.")
    
    return blackAndWhiteImage


def salt(img, n):
    img_array = np.array(img)  # Convert PIL image to NumPy array
    for k in range(n):
        i = int(np.random.random() * img_array.shape[1])
        j = int(np.random.random() * img_array.shape[0])
        if img_array.ndim == 2:
            img_array[j, i] = 255
        elif img_array.ndim == 3:
            img_array[j, i, 0] = 255
            img_array[j, i, 1] = 255
            img_array[j, i, 2] = 255
    return img_array  

def preprocess(file_path):
    img = crop_image(file_path)
    width = 315
    height = 315
    lead_images = []
    I_img   = img.crop((120.5, 0.5, width + 120.5 , 0.5 + height)).convert('L') # Converting Images to Grayscale
    II_img  = img.crop((120.5, 315.5, width + 120.5 , 315.5+ height)).convert('L')
    III_img = img.crop((120.5, 630.5, width + 120.5 , 630.5+ height)).convert('L')
    aVR_img = img.crop((672.5, 0.5, width + 672.5 , 0.5 + height)).convert('L')
    aVL_img = img.crop((672.5, 315.5, width + 672.5 , 315.5+ height)).convert('L')
    aVF_img = img.crop((672.5, 630.5, width + 672.5 , 630.5+ height)).convert('L')
    V1_img  = img.crop((1133.5, 0.5, width + 1133.5 , 0.5+ height)).convert('L')
    V2_img  = img.crop((1133.5, 315.5, width + 1133.5 , 315.5+ height)).convert('L')
    V3_img  = img.crop((1133.5, 630.5, width + 1133.5 , 630.5+ height)).convert('L')
    V4_img  = img.crop((1639.5, 0.5, width + 1639.5 , 0.5 + height)).convert('L')
    V5_img  = img.crop((1639.5, 0.5, width + 1639.5 , 0.5+ height)).convert('L')
    V6_img  = img.crop((1639.5, 630.5, width + 1639.5 , 630.5+ height)).convert('L')
    lead_images.append(I_img)
    lead_images.append(II_img)
    lead_images.append(III_img)
    lead_images.append(aVR_img)
    lead_images.append(aVL_img)
    lead_images.append(aVF_img)
    lead_images.append(V1_img)
    lead_images.append(V2_img)
    lead_images.append(V3_img)
    lead_images.append(V4_img)
    lead_images.append(V5_img)
    lead_images.append(V6_img)
    #still more preprocessing to do
    image_size = 100
    data = []
    for i in range(len(lead_images)):
        img_array = bg_remov(lead_images[i])
        new_img_array = cv2.resize(img_array, (image_size, image_size)) 
        data.append(new_img_array)
    return data

def process_image(image_path):
    lead_predictions = []
    images = preprocess(image_path)
    for lead_image in images:
        img_array = np.array(lead_image)
        img_array = np.expand_dims(img_array, axis=0) 
        prediction = model.predict(img_array)
        
        lead_predictions.append(prediction) # Assuming the predictions are numpy arrays
    combined_prediction = np.mean(lead_predictions, axis=0)
    return np.argmax(combined_prediction)

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        message = None
        num = None
        if 'image' not in request.files:
            return jsonify({'message': 'No image selected'})
        file = request.files['image']
        if file.filename == '':
            return jsonify({'message': 'No image selected'})
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            image_url = '/' + file_path  # Assuming the app is running on the root path
            num = process_image(file_path)
            if num == 0 or num == 3:
                message = "Abnormal heart-beat,Please consult a Doctor immediately!"
            elif num == 1 :
                message = "Myocaridal Infraction,please consult a doctor immediately"
            elif num == 2:
                message = "Normal"
        
            return jsonify({'message':message,'image_url': image_url})
        else:
            return jsonify({'message': 'Invalid file type'})
        
    return render_template('heartcnn-final.html')

if __name__ == '__main__':
    app.run(debug=True)
