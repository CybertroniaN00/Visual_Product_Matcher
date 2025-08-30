import io
import os
import json
import requests
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torchvision.transforms as transforms
from torchvision import models
import webcolors
from colorthief import ColorThief

DATABASE_PATH = os.path.join(os.path.dirname(__file__), "database.json")

# --------------------------
# JSON Encoder for NumPy Types
# --------------------------
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)

# --------------------------
# Safe JSON Helpers
# --------------------------
def safe_load_json(path=DATABASE_PATH, default=None):
    if default is None:
        default = []
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump(default, f, indent=4, cls=NumpyEncoder)
        return default
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        with open(path, "w") as f:
            json.dump(default, f, indent=4, cls=NumpyEncoder)
        return default

def safe_save_json(data, path=DATABASE_PATH):
    with open(path, "w") as f:
        json.dump(data, f, indent=4, cls=NumpyEncoder)

# --------------------------
# Utility Functions
# --------------------------
def fetch_image_from_url(image_url: str) -> Image.Image:
    response = requests.get(image_url)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")

def closest_color(requested_color):
    min_colors = {}
    for name in webcolors.names():
        r_c, g_c, b_c = webcolors.name_to_rgb(name)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

# --------------------------
# Dominant Colors
# --------------------------
def get_dominant_colors(img: Image.Image, num_colors=3) -> list:
    from io import BytesIO
    img_bytes = BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    color_thief = ColorThief(img_bytes)
    palette = color_thief.get_palette(color_count=num_colors, quality=1)
    
    results = []
    for rgb in palette:
        try:
            hex_value = webcolors.rgb_to_hex(rgb)
            name = webcolors.hex_to_name(hex_value)
        except ValueError:
            name = closest_color(rgb)
        results.append({"rgb": rgb, "name": name})
    return results

# --------------------------
# Feature Extractor
# --------------------------
class FeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = torch.nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        with torch.no_grad():
            feats = self.features(x)
        return feats.view(feats.size(0), -1)

# --------------------------
# Model Setup
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet50(pretrained=True).eval().to(device)
feature_extractor = FeatureExtractor(resnet).to(device)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# --------------------------
# Main Processing Function
# --------------------------
def process_image(image_source, is_url=True):
    # Load image
    img = fetch_image_from_url(image_source) if is_url else Image.open(image_source).convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    # Category
    with torch.no_grad():
        outputs = resnet(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        top_catid = torch.argmax(probs)
    category = f"class_{top_catid.item()}"

    # Embeddings
    embeddings = feature_extractor(input_tensor).cpu().numpy().flatten().tolist()

    # Dominant colors
    colors = get_dominant_colors(img)

    # Product entry
    product_entry = {
        "category": category,
        "dominant_colors": colors,
        "embedding": embeddings
    }
    return product_entry

# --------------------------
# Similarity Search
# --------------------------
def find_similar_products(query_entry, top_n=20):
    db = safe_load_json()
    if not db:
        return []

    # Filter by category first
    filtered = [p for p in db if p["category"] == query_entry["category"]]

    # Filter by at least 2 common colors
    def color_match(p):
        query_colors = {c["name"] for c in query_entry["dominant_colors"]}
        target_colors = {c["name"] for c in p["dominant_colors"]}
        return len(query_colors & target_colors) >= 2

    filtered = [p for p in filtered if color_match(p)]
    if not filtered:
        filtered = db  # fallback if no match

    # Similarity by embeddings
    db_embeddings = np.array([p["embedding"] for p in filtered])
    query_emb = np.array(query_entry["embedding"]).reshape(1, -1)
    similarities = cosine_similarity(query_emb, db_embeddings)[0]

    top_indices = similarities.argsort()[-top_n:][::-1]
    similar_products = [filtered[i] for i in top_indices]
    return similar_products
