import os
import zipfile
import logging
from speakerProcessor import Diarization
from transcriber import Transcribe, merge_transcriptions
from writeChat import process_transcription_file
from writeChat import update_speaker_names_in_html
from extractiveSummary import get_important_sentences
from datetime import datetime

diarization_model = "pyannote/speaker-diarization-3.0"
transcription_model = "openai/whisper-large-v2"
# transcription_model = "openai/whisper-large-v3"
# transcription_model = "openai/whisper-tiny"

def process_audio_file(audio_path, num_speakers, HTML, zip_name):
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    rttm_path = f"/home/evert/Desktop/audio/{base_name}.rttm"
    output_path = f"/home/evert/Desktop/audio/{base_name}"
    html_path = f"/home/evert/Desktop/audio/{base_name}.html"
    summary_path = f"/home/evert/Desktop/audio/{base_name}_sum.txt"

    try:
        Diarization(audio_path, rttm_path, diarization_model, num_speakers)
        Transcribe(audio_path, rttm_path, transcription_model, output_path)
        merge_transcriptions(output_path, output_path)

        if HTML == 1:
            html_content = process_transcription_file(output_path + ".json")
            with open(html_path, 'w') as file:
                file.write(html_content)
            logging.info(f"HTML construction done, file written to: {html_path}")

        file_names = [output_path + ".json", output_path + ".txt", output_path + "_elaborate.txt", html_path]
        with zipfile.ZipFile(zip_name + ".zip", 'a') as myzip:
            for file_name in file_names:
                base_file_name = os.path.basename(file_name)
                myzip.write(file_name, arcname="".join([base_name, "/", base_file_name]))
            logging.info(f'Created {zip_name}.zip containing {file_names}')

        if SUM == 1:
            get_important_sentences(output_path + ".txt", summary_path, prob_threshold=0.8)
    except Exception as e:
        logging.error(f"Error processing {audio_path}: {str(e)}")

HTML = 1
SUM = 0

input_file = 'transcription_input.txt'
current_datetime = datetime.now()
time_stamp = current_datetime.strftime("%Y-%m-%d-%H:%M")
zip_name = "/home/evert/Desktop/audio/" + time_stamp

logging.basicConfig(filename="".join([time_stamp,'.log']), level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

with open(input_file, 'r') as file:
    for line in file:
        line = line.strip()
        if line in ('HTML', 'SUM'):
            HTML = 1 if line == 'HTML' else HTML
            SUM = 1 if line == 'SUM' else SUM
            continue

        if not line or line.startswith('#'):
            continue

        parts = line.split(', ')
        audio_path = parts[0]
        num_speakers = None if len(parts) == 1 else int(parts[1])
        try:
            process_audio_file(audio_path, num_speakers, HTML, zip_name)
        except Exception as e:
            logging.error(f"Failed to process {audio_path}: {e}")


