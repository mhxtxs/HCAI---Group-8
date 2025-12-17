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
import uuid
from werkzeug.utils import secure_filename
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler





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

PUBLIC_IMAGE_DIR = os.path.join(app.static_folder, "training_images")
os.makedirs(PUBLIC_IMAGE_DIR, exist_ok=True)



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


ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def _iter_training_image_paths():
    candidate_dirs = [
        IMAGE_DIR,
    ]
    seen = set()
    for d in candidate_dirs:
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            ext = os.path.splitext(fn)[1].lower()
            if ext not in ALLOWED_IMAGE_EXTS:
                continue
            stem = os.path.splitext(fn)[0].replace("_", " ").lower()
            if base_class(stem) == "Unknown":
                continue
            path = os.path.join(d, fn)
            if path in seen:
                continue
            seen.add(path)
            yield path

def build_dataset_from_images():
    paths = sorted(list(_iter_training_image_paths()))
    points = []
    for i, path in enumerate(paths, start=1):
        label = os.path.splitext(os.path.basename(path))[0].replace("_", " ")
        with open(path, "rb") as f:
            feats = image_to_features(io.BytesIO(f.read()))
        if feats is None:
            continue
        url_path = "/" + path.replace("\\", "/")
        points.append({
            "id": str(i),
            "label": label,
            "image": url_path,
            "features": feats.tolist(),
            "coords": [0.0, 0.0],
        })
    return points

def _looks_like_iris_measurements(dataset):
    try:
        feats = np.asarray([p.get("features", []) for p in dataset], dtype=float)
        if feats.size == 0:
            return True
        return float(np.nanmax(feats)) > 1.2
    except Exception:
        return True

def load_dataset():
    dataset = None
    if os.path.exists(DATASET_FILE):
        try:
            with open(DATASET_FILE, "r") as f:
                dataset = json.load(f)
        except Exception:
            dataset = None

    if not dataset:
        rebuilt = build_dataset_from_images()
        if rebuilt:
            save_dataset(rebuilt)
            return rebuilt
        return [dict(p) for p in DEFAULT_DATASET]

    if _looks_like_iris_measurements(dataset):
        rebuilt = build_dataset_from_images()
        if rebuilt:
            save_dataset(rebuilt)
            return rebuilt

    return dataset



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
    a = np.asarray(p1, dtype=float)
    b = np.asarray(p2, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Vector shape mismatch")
    diff = np.abs(a - b)
    if metric == "manhattan":
        return float(diff.sum())
    if metric == "minkowski":
        return float((diff ** p).sum() ** (1.0 / p))
    return float(np.sqrt(((a - b) ** 2).sum()))


def base_class(label):
    s = label.lower()
    if "setosa" in s:
        return "Setosa"
    if "versicolor" in s:
        return "Versicolor"
    if "virginica" in s:
        return "Virginica"
    return "Unknown"


def knn_predict(query_vec, dataset, k, metric, p, vector_key="coords"):
    distances = []
    for point in dataset:
        vec = point.get(vector_key)
        if vec is None:
            continue
        d = compute_distance(query_vec, vec, metric, p)
        distances.append((d, point["label"]))
    distances.sort(key=lambda x: x[0])
    nearest = distances[: max(1, min(int(k), len(distances)))]
    votes = {}
    for _, label in nearest:
        c = base_class(label)
        votes[c] = votes.get(c, 0) + 1
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

def fit_projector(dataset):
    feats = np.asarray([p.get("features", []) for p in dataset], dtype=float)
    if len(dataset) < 2:
        return None, np.zeros((len(dataset), 2), dtype=float)

    y = [base_class(p.get("label", "")) for p in dataset]
    classes = sorted(set(y))

    if len(classes) >= 2 and len(dataset) >= 3:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(feats)
        lda = LinearDiscriminantAnalysis(n_components=2)
        coords = lda.fit_transform(Xs, y)
        return ("lda", scaler, lda), coords

    pca = PCA(n_components=2)
    coords = pca.fit_transform(feats)
    return ("pca", None, pca), coords


def project_query(projector, query_features):
    if projector is None:
        return [0.0, 0.0]

    kind, scaler, model = projector
    q = np.asarray(query_features, dtype=float).reshape(1, -1)

    if kind == "lda":
        q = scaler.transform(q)
        return model.transform(q)[0].tolist()

    return model.transform(q)[0].tolist()



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
    projector, coords = fit_projector(dataset)
    for p, c in zip(dataset, coords):
        p["coords"] = c.tolist()
        query_coords = project_query(projector, features)

    



    # READ FRONTEND PARAMETERS (optional, but wired)
    k = int(request.form.get("k", 5))
    metric = request.form.get("metric", "euclidean")
    mink_p = float(request.form.get("p", 2))

    # --- SIMILARITY CALCULATION ---

    distances = [
        compute_distance(features, p["features"], metric, mink_p)
        for p in dataset
        ]
    closest_d = min(distances)
    similarity_score = max(0, 1 - (closest_d / (closest_d + 1))) * 100
    similarity_score = round(similarity_score, 2)

    # --- KNN PREDICTION ---
    prediction = knn_predict(features, dataset, k, metric, mink_p, vector_key="features")

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


def _next_dataset_id(dataset):
    ids = []
    for p in dataset:
        try:
            ids.append(int(p.get("id")))
        except:
            pass
    return (max(ids) if ids else 0) + 1

def _fit_pca_on_dataset(dataset):
    feats = np.asarray([p.get("features", []) for p in dataset], dtype=float) if dataset else np.zeros((0, 4))
    if len(dataset) < 2:
        return None, np.zeros((len(dataset), 2), dtype=float)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(feats)
    return pca, coords




def _dataset_points_for_frontend(dataset):
    projector, coords = fit_projector(dataset)

    out = []
    for p, c in zip(dataset, coords):
        pid = p.get("id")
        try:
            pid_num = int(pid)
        except:
            pid_num = pid

        out.append({
            "id": pid_num,
            "label": p.get("label", ""),
            "x": float(c[0]),
            "y": float(c[1]),
            "source": p.get("source", "Training data"),
            "inTraining": True,
            "image": p.get("image", "")
        })
    return out






@app.route("/api/dataset", methods=["GET"])
def api_dataset():
    dataset = load_dataset()
    return jsonify({"dataset": _dataset_points_for_frontend(dataset)})

@app.route("/api/dataset/add", methods=["POST"])
def api_dataset_add():
    if "file" not in request.files:
        return jsonify({"error": "Missing file"}), 400

    file = request.files["file"]
    img_bytes = file.read()
    if not img_bytes:
        return jsonify({"error": "Empty file"}), 400

    label = (request.form.get("label") or "").strip()
    if not label:
        label = os.path.splitext(file.filename or "user_upload")[0].replace("_", " ").strip()

    if base_class(label) == "Unknown":
        return jsonify({"error": "Label must include setosa, versicolor, or virginica."}), 400

    feats = image_to_features(io.BytesIO(img_bytes))
    if feats is None:
        return jsonify({"error": "Invalid image"}), 400

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        ext = ".jpg"

    safe_label = secure_filename(label.replace(" ", "_").lower()) or "user_upload"
    filename = f"{safe_label}_{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(PUBLIC_IMAGE_DIR, filename)
    with open(save_path, "wb") as f:
        f.write(img_bytes)

    image_url = f"/static/training_images/{filename}"

    dataset = load_dataset()
    new_id = _next_dataset_id(dataset)

    dataset.append({
        "id": str(new_id),
        "label": label,
        "image": image_url,
        "features": feats.tolist(),
        "coords": [0.0, 0.0]
    })

    save_dataset(dataset)

    return jsonify({
        "added": {"id": new_id, "label": label, "image": image_url},
        "dataset": _dataset_points_for_frontend(dataset)
    })

@app.route("/api/dataset/remove", methods=["POST"])
def api_dataset_remove():
    payload = request.get_json() or {}
    rid = payload.get("id")
    if rid is None:
        return jsonify({"error": "Missing id"}), 400

    dataset = load_dataset()
    rid_str = str(rid)

    removed = None
    new_ds = []
    for p in dataset:
        if str(p.get("id")) == rid_str and removed is None:
            removed = p
        else:
            new_ds.append(p)

    if removed is None:
        return jsonify({"error": "Not found"}), 404

    img = removed.get("image", "")
    if isinstance(img, str) and img.startswith("/static/training_images/"):
        fn = img.split("/static/training_images/", 1)[1]
        fp = os.path.join(PUBLIC_IMAGE_DIR, fn)
        if os.path.isfile(fp):
            try:
                os.remove(fp)
            except:
                pass

    save_dataset(new_ds)
    return jsonify({"removed": rid, "dataset": _dataset_points_for_frontend(new_ds)})



if __name__ == "__main__":
    app.run(debug=True)
