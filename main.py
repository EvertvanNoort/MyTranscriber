import os
from speakerProcessor import Diarization
from transcriber import Transcribe, merge_transcriptions
from writeChat import process_transcription_file
from writeChat import update_speaker_names_in_html

diarization_model = "pyannote/speaker-diarization-3.0"
transcription_model = "openai/whisper-large-v3"
language = "english"

# Function to process a single audio file
def process_audio_file(audio_path, num_speakers):
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    rttm_path = f"/home/evert/Desktop/audio/{base_name}.rttm"
    output_path = f"/home/evert/Desktop/audio/{base_name}_transcript.json"
    html_path = f"/home/evert/Desktop/audio/{base_name}.html"

    Diarization(audio_path, rttm_path, diarization_model, num_speakers)
    Transcribe(audio_path, rttm_path, transcription_model, output_path, language)
    merge_transcriptions(output_path, output_path)

    # speakers = {f"SPEAKER_{str(i).zfill(2)}": f"Speaker {i+1}" for i in range(num_speakers)}

    html_content = process_transcription_file(output_path)#, speakers)

    # Write the result to an HTML file
    with open(html_path, 'w') as file:
        file.write(html_content)

# Read the input file and process each audio file listed
with open('transcription_input.txt', 'r') as file:
    for line in file:
        line = line.strip()
        if not line or line.startswith('#'):
            continue  # Skip empty lines and lines starting with '#'

        parts = line.split(', ')
        audio_path = parts[0]
        num_speakers = None if len(parts) == 1 else int(parts[1])
        process_audio_file(audio_path, num_speakers)

# Example usage
html_file_path = '/home/evert/Desktop/audio/Altman.html'
name_mapping = {
    'SPEAKER_00': 'Sam Altman',
    'SPEAKER_01': 'Elevate Host',
    # Add more mappings as needed
}
update_speaker_names_in_html(html_file_path, name_mapping)
