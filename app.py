import os
import subprocess
from flask import Flask, request, jsonify
from flask import send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from flask import render_template

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    directory = '/home/evert/Desktop/audio/'  # Adjust to your static files directory path
    # print(directory, filename)
    return send_from_directory(directory, filename, as_attachment=True)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audioFile' not in request.files:
        return jsonify({"error": "No file part found"}), 400
    file = request.files['audioFile']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Construct the input file for run.sh
    input_file_path = 'transcription_input.txt'
    with open(input_file_path, 'w') as input_file:
        input_file.write(filepath + '\n')

    try:
        subprocess.run(["bash", "./run.sh"], check=True)
        output_filename = f"{os.path.splitext(filename)[0]}.zip" # Example output filename, adjust accordingly
        return jsonify({"message": "File successfully uploaded and processed, script executed", "filePath": output_filename})

        # return jsonify({"message": "File successfully uploaded and processed, script executed"})
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Script execution failed: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
