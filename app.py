from flask import Flask, request, render_template, redirect, url_for
import os
from backend.main import process_image, find_similar_products

UPLOAD_FOLDER = "frontend/static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder="frontend/templates", static_folder="frontend/static")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --------------------------
# Routes
# --------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle URL
        img_url = request.form.get("image_url")
        file = request.files.get("image_file")

        if img_url:
            query_entry = process_image(img_url, is_url=True)
            display_path = img_url
        elif file:
            save_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(save_path)
            query_entry = process_image(save_path, is_url=False)
            display_path = url_for("static", filename=f"uploads/{file.filename}")
        else:
            return render_template("index.html", error="Please provide an image or URL")

        results = find_similar_products(query_entry, top_n=20)
        return render_template("results.html", query_img=display_path, results=results)
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0')
