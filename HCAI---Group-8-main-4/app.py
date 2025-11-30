import os
import io
import json
import uuid
import math
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
from sklearn.decomposition import PCA

# -----------------------------------------------------------
# INITIAL SETUP (IMPORTANT CONFIG)
# -----------------------------------------------------------
app = Flask(
    __name__,
    template_folder="templates",     # your index.html is here
    static_folder="static"           # static assets (training images)
)
CORS(app)

DATASET_FILE = "dataset.json"
IMAGE_DIR = "static/training_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# -----------------------------------------------------------
# DEFAULT DATASET
# -----------------------------------------------------------
DEFAULT_DATASET = [
    {
        "id": "1",
        "label": "iris setosa",
        "image": "",
        "features": [5.1, 3.5, 1.4, 0.2],
        "coords": [0.0, 0.0],
    },
    {
        "id": "2",
        "label": "alcea setosa",
        "image": "",
        "features": [4.9, 3.0, 1.4, 0.2],
        "coords": [0.0, 0.0],
    },
    {
        "id": "3",
        "label": "echeveria setosa",
        "image": "",
        "features": [4.7, 3.2, 1.3, 0.2],
        "coords": [0.0, 0.0],
    },
    {
        "id": "4",
        "label": "oxalis versicolor",
        "image": "",
        "features": [6.2, 2.8, 4.8, 1.8],
        "coords": [0.0, 0.0],
    },
    {
        "id": "5",
        "label": "sumpfiris iris versicolor",
        "image": "",
        "features": [6.0, 2.7, 4.5, 1.5],
        "coords": [0.0, 0.0],
    },
    {
        "id": "6",
        "label": "clematis versicolor",
        "image": "",
        "features": [5.9, 2.9, 4.2, 1.5],
        "coords": [0.0, 0.0],
    },
    {
        "id": "7",
        "label": "iris virginica",
        "image": "",
        "features": [7.2, 3.0, 6.0, 2.0],
        "coords": [0.0, 0.0],
    },
    {
        "id": "8",
        "label": "mertensia virginica",
        "image": "",
        "features": [6.5, 3.0, 5.5, 2.1],
        "coords": [0.0, 0.0],
    },
    {
        "id": "9",
        "label": "itea virginica",
        "image": "",
        "features": [6.7, 3.1, 5.6, 2.4],
        "coords": [0.0, 0.0],
    },
]

# -----------------------------------------------------------
# LOAD / SAVE DATASET
# -----------------------------------------------------------
def load_dataset():
    if not os.path.exists(DATASET_FILE):
        return [dict(p) for p in DEFAULT_DATASET]

    try:
        with open(DATASET_FILE, "r") as f:
            data = json.load(f)
        return data if data else [dict(p) for p in DEFAULT_DATASET]
    except:
        return [dict(p) for p in DEFAULT_DATASET]


def save_dataset(dataset):
    with open(DATASET_FILE, "w") as f:
        json.dump(dataset, f, indent=4)

# -----------------------------------------------------------
# IMAGE FEATURE EXTRACTION
# -----------------------------------------------------------
def image_to_features(image_bytes):
    try:
        img = Image.open(image_bytes).convert("RGB")
        img = img.resize((64, 64))
        arr = np.array(img) / 255.0

        # Basic meaningful features:
        mean_r = arr[:, :, 0].mean()
        mean_g = arr[:, :, 1].mean()
        mean_b = arr[:, :, 2].mean()
        brightness = arr.mean()

        return np.array([mean_r, mean_g, mean_b, brightness])
    except Exception as e:
        print("IMAGE FEATURE ERROR:", e)
        return None

# -----------------------------------------------------------
# DISTANCE METRICS
# -----------------------------------------------------------
def compute_distance(p1, p2, metric="euclidean", p=2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]

    if metric == "manhattan":
        return abs(dx) + abs(dy)
    if metric == "minkowski":
        return (abs(dx)**p + abs(dy)**p)**(1/p)

    return math.sqrt(dx*dx + dy*dy)

# -----------------------------------------------------------
# KNN PREDICTION
# -----------------------------------------------------------
def base_class(label):
    s = label.lower()
    if "setosa" in s: return "Setosa"
    if "versicolor" in s: return "Versicolor"
    if "virginica" in s: return "Virginica"
    return "Unknown"


def knn_predict(query_vec, dataset, k, metric, p):
    distances = []
    for point in dataset:
        d = compute_distance(query_vec, point["coords"], metric, p)
        distances.append((d, point["label"]))

    distances.sort(key=lambda x: x[0])
    nearest = distances[:k]

    votes = {}
    for _, label in nearest:
        c = base_class(label)
        votes[c] = votes.get(c, 0) + 1

    return max(votes.items(), key=lambda x: x[1])[0]

# -----------------------------------------------------------
# HOMEPAGE ROUTE  ✔ FIXED
# -----------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -----------------------------------------------------------
# API: UPLOAD IMAGE (frontend expects this)
# -----------------------------------------------------------
@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img_bytes = file.read()
    features = image_to_features(io.BytesIO(img_bytes))

    if features is None:
        return jsonify({"error": "Invalid image"}), 400

    dataset = load_dataset()
    feats = np.array([p["features"] for p in dataset])
    all_feats = np.vstack([feats, features])

    pca = PCA(n_components=2)
    coords = pca.fit_transform(all_feats)

    return jsonify({
        "coords": coords[-1].tolist(),
        "message": "Image processed"
    })

# -----------------------------------------------------------
# API: PREDICT
# -----------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Missing file"}), 400

    file = request.files["file"]
    img_bytes = file.read()
    features = image_to_features(io.BytesIO(img_bytes))

    if features is None:
        return jsonify({"error": "Invalid image"}), 400

    dataset = load_dataset()
    data_feats = np.array([p["features"] for p in dataset])
    all_feats = np.vstack([data_feats, features])

    pca = PCA(n_components=2)
    coords = pca.fit_transform(all_feats)

    # update dataset coords
    for p, c in zip(dataset, coords[:-1]):
        p["coords"] = c.tolist()

    query_coords = coords[-1].tolist()

    k = int(request.form.get("k", 5))
    metric = request.form.get("metric", "euclidean")
    mink_p = float(request.form.get("p", 2))

    prediction = knn_predict(query_coords, dataset, k, metric, mink_p)

    return jsonify({
        "prediction": f"The flower is most likely **{prediction}**.",
        "class": prediction,
        "coords": query_coords
    })

# -----------------------------------------------------------
# API: ADD IMAGE TO DATASET
# -----------------------------------------------------------
@app.route("/api/add_image", methods=["POST"])
def add_image():
    label = request.form.get("label", "unknown_flower")
    file = request.files.get("file")

    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    img_bytes = file.read()
    features = image_to_features(io.BytesIO(img_bytes))

    if features is None:
        return jsonify({"error": "Invalid image"}), 400

    filename = f"{uuid.uuid4().hex[:12]}.jpg"
    path = os.path.join(IMAGE_DIR, filename)
    with open(path, "wb") as f:
        f.write(img_bytes)

    dataset = load_dataset()
    dataset.append({
        "id": str(uuid.uuid4()),
        "label": label,
        "image": path,
        "features": features.tolist(),
        "coords": [0, 0],
    })

    feats = np.array([p["features"] for p in dataset])
    coords = PCA(n_components=2).fit_transform(feats)

    for p, c in zip(dataset, coords):
        p["coords"] = c.tolist()

    save_dataset(dataset)

    return jsonify({"message": "Image added"})

# -----------------------------------------------------------
# API: CHAT
# -----------------------------------------------------------
@app.route("/api/chat", methods=["POST"])
def chat():
    msg = (request.get_json() or {}).get("message", "").lower()

    if "how" in msg and "work" in msg:
        return jsonify({
            "reply": "I compare your flower with known flowers using the KNN algorithm."
        })

    if "setosa" in msg:
        return jsonify({"reply": "Setosa flowers are small and compact."})

    if "versicolor" in msg:
        return jsonify({"reply": "Versicolor flowers have medium-sized petals."})

    if "virginica" in msg:
        return jsonify({"reply": "Virginica flowers have long petals."})

    return jsonify({
        "reply": "FlowerLens analyses flower similarity using KNN. Upload a flower and ask more!"
    })

# -----------------------------------------------------------
# RUN SERVER
# -----------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)