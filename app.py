from flask import Flask, render_template, request, redirect, url_for, jsonify
import numpy as np
import json
import io
import tensorflow as tf
import os

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("models/my_keras_model.keras")

# Labels
label = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Background_without_leaves',
    'Blueberry___healthy',
    'Cherry___Powdery_mildew',
    'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight',
    'Corn___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Load JSON disease info
with open("plant_disease.json", "r") as file:
    plant_disease = json.load(file)


@app.route('/')
def home():
    return render_template('home.html')


def extract_features(image_file):
    image_file.seek(0)
    image_bytes = io.BytesIO(image_file.read())
    image = tf.keras.utils.load_img(image_bytes, target_size=(224,224))
    feature = tf.keras.utils.img_to_array(image)
    feature = feature / 255.0
    return np.expand_dims(feature, axis=0)


def model_predict(image):
    img = extract_features(image)
    prediction = model.predict(img)
    prediction_label = label[prediction.argmax()]
    return prediction_label


@app.route('/upload/', methods=['POST', 'GET'])
def uploadimage():
    if request.method == "POST":
        image = request.files['img']
        prediction = model_predict(image)

        disease_info = next(
            (item for item in plant_disease if item["name"] == prediction),
            {"cause": "Not found", "cure": "Not found"}
        )

        return jsonify(
            success=True,
            prediction=prediction,
            disease_info=disease_info
        )
    return redirect('/')


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)