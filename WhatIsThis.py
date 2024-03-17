from flask import Flask, request, jsonify
import subprocess
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# app = Flask(__name__)
@app.route('/')
def index():
    return render_template('start.html')


@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.json
    audio_path = data.get('audioPath')
    html_output = data.get('htmlOutput')

    # Modify the input file based on the form data
    with open('transcription_input.txt', 'w') as file:
        file.write(f"HTML\n" if html_output else "")
        file.write(f"{audio_path}\n")

    # Execute the shell script
    subprocess.run(["bash", "./run.sh"])

    return jsonify({"message": "Transcription initiated successfully."})

if __name__ == "__main__":
    app.run(debug=True)
