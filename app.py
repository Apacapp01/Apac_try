import os
import cv2
import base64
import pickle
import numpy as np
import mediapipe as mp
import tflite_runtime.interpreter as tflite

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image

# ---------------- APP SETUP ----------------
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "static/images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# ---------------- GLOBAL CACHE ----------------
models = {}       # lazy-loaded TFLite interpreters
scaler = None
labels = None
hands = None

# ---------------- LAZY LOAD COMMON RESOURCES ----------------
def load_common_resources():
    global scaler, labels, hands

    if scaler is None:
        with open("models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

    if labels is None:
        with open("models/sign_language_features.pkl", "rb") as f:
            labels = pickle.load(f)["labels"]

    if hands is None:
        hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        )

# ---------------- LOAD TFLITE MODEL ----------------
def load_model(model_type):
    if model_type in models:
        return models[model_type]

    model_paths = {
        "live": "models/final_model.tflite",
        "letter": "models/L_model.tflite",
        "number": "models/N_model.tflite"
    }

    interpreter = tflite.Interpreter(model_path=model_paths[model_type])
    interpreter.allocate_tensors()

    models[model_type] = interpreter
    return interpreter

# ---------------- UTILS ----------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_keypoints(image_np):
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    keypoints = np.zeros(126, dtype=np.float32)

    if results.multi_hand_landmarks:
        idx = 0
        for hand in results.multi_hand_landmarks:
            for lm in hand.landmark:
                if idx >= 126:
                    break
                keypoints[idx:idx+3] = (lm.x, lm.y, lm.z)
                idx += 3

    return scaler.transform(keypoints.reshape(1, -1)).astype(np.float32)

def predict_tflite(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])
    return output

# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload_image", methods=["POST"])
def upload_image():
    load_common_resources()

    file = request.files.get("file")
    model_type = request.form.get("type", "live")

    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    try:
        img = Image.open(path).convert("RGB")
        img_np = np.array(img)

        keypoints = extract_keypoints(img_np)
        interpreter = load_model(model_type)

        pred = predict_tflite(interpreter, keypoints)
        label = labels[int(np.argmax(pred))]

        return jsonify({
            "success": True,
            "model": model_type,
            "prediction": label
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/process_frame", methods=["POST"])
def process_frame():
    load_common_resources()
    interpreter = load_model("live")

    try:
        data = request.json["image"].split(",")[1]
        frame = cv2.imdecode(
            np.frombuffer(base64.b64decode(data), np.uint8),
            cv2.IMREAD_COLOR
        )

        keypoints = extract_keypoints(frame)
        pred = predict_tflite(interpreter, keypoints)

        return jsonify({
            "has_hands": True,
            "prediction": labels[int(np.argmax(pred))]
        })

    except Exception:
        return jsonify({"has_hands": False})

# ---------------- START ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
