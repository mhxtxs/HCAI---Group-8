import os
import io
import json
import math
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
from sklearn.decomposition import PCA
import base64
import cv2


app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)
CORS(app)

DATASET_FILE = "dataset.json"
IMAGE_DIR = "static/training_images"
HEATMAP_DIR = "static/heatmaps"

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)


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


FLOWER_INFO = {
    "Setosa": {
        "summary": "Small compact flower.\nFound mostly in cool climates."
    },
    "Versicolor": {
        "summary": "Medium-sized violet flower.\nOften grows in marshy environments."
    },
    "Virginica": {
        "summary": "Large blue-purple petals.\nCommon in wetlands and moist forests."
    }
}


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


def image_to_features(image_bytes):
    try:
        img = Image.open(image_bytes).convert("RGB")
        img = img.resize((64, 64))
        arr = np.array(img) / 255.0

        mean_r = arr[:, :, 0].mean()
        mean_g = arr[:, :, 1].mean()
        mean_b = arr[:, :, 2].mean()
        brightness = arr.mean()

        return np.array([mean_r, mean_g, mean_b, brightness])
    except Exception as e:
        print("IMAGE FEATURE ERROR:", e)
        return None

def compute_distance(p1, p2, metric="euclidean", p=2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]

    if metric == "manhattan":
        return abs(dx) + abs(dy)

    if metric == "minkowski":
        return (abs(dx) ** p + abs(dy) ** p) ** (1 / p)

    return math.sqrt(dx * dx + dy * dy)


def base_class(label):
    s = label.lower()
    if "setosa" in s:
        return "Setosa"
    if "versicolor" in s:
        return "Versicolor"
    if "virginica" in s:
        return "Virginica"
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

    # return class name: "Setosa" / "Versicolor" / "Virginica"
    return max(votes.items(), key=lambda x: x[1])[0]

def generate_heatmap(img_bytes):
    """
    Creates a saliency-style gradient heatmap.
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((256, 256))
    arr = np.array(img).astype(np.float32)

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    grad = np.sqrt(gx ** 2 + gy ** 2)

    grad_norm = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    heatmap = cv2.applyColorMap(grad_norm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(arr.astype(np.uint8), 0.5, heatmap, 0.5, 0)

    output_path = os.path.join(HEATMAP_DIR, "last_heatmap.jpg")
    cv2.imwrite(output_path, overlay)

    return "/static/heatmaps/last_heatmap.jpg"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/model-card")
@app.route("/model-card.html")
def model_card():
    return render_template("model-card.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Missing file"}), 400

    file = request.files["file"]
    img_bytes = file.read()

    # ENCODE ORIGINAL IMAGE IN BASE64
    original_b64 = base64.b64encode(img_bytes).decode("utf-8")
    original_data_uri = "data:image/jpeg;base64," + original_b64

    # IMAGE TO FEATURES
    features = image_to_features(io.BytesIO(img_bytes))
    if features is None:
        return jsonify({"error": "Invalid image"}), 400

    dataset = load_dataset()
    data_feats = np.array([p["features"] for p in dataset])
    all_feats = np.vstack([data_feats, features])

    # PCA → 2D COORDS
    pca = PCA(n_components=2)
    coords = pca.fit_transform(all_feats)

    # UPDATE TRAINING COORDS
    for p, c in zip(dataset, coords[:-1]):
        p["coords"] = c.tolist()

    query_coords = coords[-1].tolist()

    # READ FRONTEND PARAMETERS (optional, but wired)
    k = int(request.form.get("k", 5))
    metric = request.form.get("metric", "euclidean")
    mink_p = float(request.form.get("p", 2))

    # --- SIMILARITY CALCULATION ---
    distances = [
        compute_distance(query_coords, p["coords"], metric, mink_p)
        for p in dataset
    ]
    closest_d = min(distances)
    similarity_score = max(0, 1 - (closest_d / (closest_d + 1))) * 100
    similarity_score = round(similarity_score, 2)

    # --- KNN PREDICTION ---
    prediction = knn_predict(query_coords, dataset, k, metric, mink_p)

    # --- HEATMAP ---
    heatmap_url = generate_heatmap(img_bytes)

    # SHORT SUMMARY (2 lines) based on predicted class
    summary = FLOWER_INFO.get(
        prediction,
        {"summary": "No description available.\nUnknown habitat."}
    )["summary"]

    return jsonify({
        "prediction": f"The flower is likely {prediction}.",
        "class": prediction,
        "coords": query_coords,
        "similarity": similarity_score,
        "original_image": original_data_uri,
        "heatmap_image": heatmap_url,
        "summary": summary
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    msg = (request.get_json() or {}).get("message", "").lower()

    if "how" in msg and "work" in msg:
        return jsonify({
            "reply": "I compare your flower with known flowers using the KNN algorithm."
        })

    if "setosa" in msg:
        return jsonify({"reply": "Setosa is usually small and compact, often found in cool climates."})

    if "versicolor" in msg:
        return jsonify({"reply": "Versicolor is a medium-sized violet flower that often grows in marshy areas."})

    if "virginica" in msg:
        return jsonify({"reply": "Virginica has large blue-purple petals and likes wetland habitats."})

    return jsonify({
        "reply": "You can ask me about how KNN works or about Setosa, Versicolor, or Virginica."
    })


if __name__ == "__main__":
    app.run(debug=True)
