from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
from detect_plate4 import process_image  # 아래에서 함수화할 예정

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
RESULT_FOLDER = './res9'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    result_image_path, result_text = process_image(file_path)
    return jsonify({'text': result_text, 'image': os.path.basename(result_image_path)})

@app.route('/image/<filename>')
def serve_image(filename):
    return send_file(os.path.join(RESULT_FOLDER, filename), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)

