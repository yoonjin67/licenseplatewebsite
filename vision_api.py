# vision_api.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from vision_server import process_image
from werkzeug.utils import secure_filename
from pathlib import Path

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = Path(__file__).resolve().parent / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    filename = secure_filename(file.filename)
    filepath = UPLOAD_FOLDER / filename
    file.save(filepath)

    result_path, label = process_image(str(filepath))
    return jsonify({
        "result_path": f"/api/image/{Path(result_path).name}",
        "label": label
    })

@app.route('/api/image/<filename>')
def get_image(filename):
    image_path = Path("res_web") / filename
    if not image_path.exists():
        return f"{filename} not found", 404
    return send_file(str(image_path), mimetype='image/jpeg')

# ✅ SAMPLE API 추가
@app.route('/api/samples', methods=['GET'])
def get_sample_images():
    image_dir = Path("images")
    if not image_dir.exists():
        return jsonify([])

    samples = []
    for path in sorted(image_dir.glob("*.jpg")):
        result_path, label = process_image(str(path))
        samples.append({
            "filename": path.name,
            "label": label,
            "result_url": f"/api/image/{Path(result_path).name}"
        })
    return jsonify(samples)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5050)

