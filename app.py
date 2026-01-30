import os
import cv2
import base64
import pickle
import numpy as np
import mediapipe as mp
import tensorflow as tf

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image

# ---------------- BASIC APP SETUP ----------------
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "static/images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# ---------------- TF CPU CONFIG (RENDER SAFE) ----------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

# ---------------- GLOBAL OBJECTS (LAZY LOADED) ----------------
models = None
scaler = None
labels = None

# ---------------- MEDIAPIPE (GLOBAL, NOT PER REQUEST) ----------------
mp_hands = mp.solutions.hands
hands_static = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)
hands_live = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---------------- LOAD MODELS ON FIRST REQUEST ----------------
@app.before_first_request
def load_all_models():
    global models, scaler, labels

    from tensorflow.keras.models import load_model

    models = {
        "live": load_model("models/final_model.keras", compile=False),
        "letter": load_model("models/L_model.h5", compile=False),
        "number": load_model("models/N_model.h5", compile=False),
    }

    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("models/sign_language_features.pkl", "rb") as f:
        labels = pickle.load(f)["labels"]

    print("✅ Models loaded successfully")

# ---------------- UTILITIES ----------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img):
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def extract_keypoints(image_np):
    results = hands_static.process(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    keypoints = np.zeros(126)

    if results.multi_hand_landmarks:
        idx = 0
        for hand in results.multi_hand_landmarks:
            for lm in hand.landmark:
                if idx >= 126:
                    break
                keypoints[idx:idx+3] = [lm.x, lm.y, lm.z]
                idx += 3

    return keypoints.reshape(1, -1)

# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return "Backend running 🚀"

@app.route("/upload_image", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    img = Image.open(path)
    img_np = np.array(img)

    keypoints = extract_keypoints(img_np)
    keypoints_scaled = scaler.transform(keypoints)

    pred = models["live"].predict(keypoints_scaled, verbose=0)
    label = labels[np.argmax(pred)]

    return jsonify({
        "success": True,
        "prediction": label
    })

@app.route("/process_frame", methods=["POST"])
def process_frame():
    data = request.json["image"].split(",")[1]
    frame = cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_COLOR)

    results = hands_live.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks:
        return jsonify({"has_hands": False})

    keypoints = extract_keypoints(frame)
    keypoints_scaled = scaler.transform(keypoints)
    pred = models["live"].predict(keypoints_scaled, verbose=0)

    return jsonify({
        "has_hands": True,
        "prediction": labels[np.argmax(pred)]
    })

# ---------------- START ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
