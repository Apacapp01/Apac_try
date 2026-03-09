import os
import base64
import time
import pickle
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from functools import lru_cache

# ---------------- APP SETUP ----------------
app = Flask(__name__)
CORS(app)

# ---------------- CONFIG ----------------
app.config['UPLOAD_FOLDER'] = 'static/images'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ---------------- TENSORFLOW SETTINGS ----------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

# ---------------- LABELS ----------------
letter_labels = {i: chr(65 + i) for i in range(26)}
number_labels = {i: str(i) for i in range(10)}

class_labels = {
    0: 'are', 1: 'did', 2: 'doing', 3: 'eat',
    4: 'going', 5: 'How', 6: 'is', 7: 'name',
    8: 'What', 9: 'Where', 10: 'Which', 11: 'you', 12: 'your'
}

# ---------------- MODEL LOADING ----------------
@lru_cache(maxsize=1)
def load_models_and_scaler():

    models = {
        'live_model': load_model('models/final_model.keras', compile=False),
        'letter_model': load_model('models/L_model.h5', compile=False),
        'number_model': load_model('models/N_model.h5', compile=False),
        'word_model': load_model('models/W_model.h5', compile=False)
    }

    with open('models/sign_language_features.pkl', 'rb') as f:
        data = pickle.load(f)
        labels = data['labels']

    with open('models/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    return models, labels, scaler


models, labels, scaler = load_models_and_scaler()

# ---------------- UTILS ----------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def preprocess_image(image, target_size=(224, 224)):

    if image.mode != 'RGB':
        image = image.convert('RGB')

    img_array = np.array(image.resize(target_size))

    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    return img_array.astype(np.float32) / 255.0


def extract_keypoints_two_hands(image_np):

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    ) as hands:

        results = hands.process(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

        keypoints = np.zeros(126, dtype=np.float32)

        if results.multi_hand_landmarks:

            idx = 0

            for hand in results.multi_hand_landmarks:

                for lm in hand.landmark:

                    if idx >= 126:
                        break

                    keypoints[idx] = lm.x
                    keypoints[idx+1] = lm.y
                    keypoints[idx+2] = lm.z
                    idx += 3

        return keypoints.reshape(1, 126)


# ---------------- ROUTES ----------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload_image', methods=['POST'])
def upload_image():

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    file.save(filepath)

    try:

        img = Image.open(filepath)
        img_array = preprocess_image(img)

        pred_letter = models['letter_model'].predict(np.expand_dims(img_array, 0), verbose=0)
        pred_number = models['number_model'].predict(np.expand_dims(img_array, 0), verbose=0)

        conf_letter = float(np.max(pred_letter))
        conf_number = float(np.max(pred_number))

        if conf_letter >= conf_number:
            prediction = letter_labels[np.argmax(pred_letter)]
        else:
            prediction = number_labels[np.argmax(pred_number)]

        return jsonify({
            "success": True,
            "prediction": prediction,
            "confidence": max(conf_letter, conf_number)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/process_frame', methods=['POST'])
def process_frame():

    data = request.get_json()

    if not data or 'image' not in data:
        return jsonify({"error": "No image provided"}), 400

    image_data = data['image'].split(',')[1]
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    keypoints_array = extract_keypoints_two_hands(frame)

    keypoints_flat = keypoints_array.reshape(1, -1)
    keypoints_scaled = scaler.transform(keypoints_flat)

    prediction = models['live_model'].predict(keypoints_scaled, verbose=0)
    predicted_label = labels[np.argmax(prediction)]

    return jsonify({
        "success": True,
        "prediction": predicted_label
    })


# ---------------- START SERVER ----------------
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 7860))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
