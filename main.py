import os
import fileinput
from speakerProcessor import Diarization
from transcriberV3 import Transcribe
from transcriberV3 import merge_transcriptions
from writeChat import process_transcription_file

audio_path = "/home/evert/Downloads/test.mp3"
rttm_path = "/home/evert/Desktop/audio/output.rttm"
output_path = "/home/evert/Desktop/audio/outputtime.txt"
html_path = "/home/evert/Desktop/audio/output.html"

diarization_model = "pyannote/speaker-diarization-3.0"
transcription_model = "openai/whisper-large-v3"

num_speakers = 2

language = "english"

Diarization(audio_path, rttm_path, diarization_model, num_speakers)
Transcribe(audio_path, rttm_path, transcription_model, output_path, language)

merge_transcriptions(output_path, output_path)

speakers = {
    "SPEAKER_00": "Sam Altman",
    "SPEAKER_01": "Elevate Host",
    # Add more mappings as needed
}

html_content = process_transcription_file(output_path, speakers)

# Write the result to an HTML file
with open(html_path, 'w') as file:
    file.write(html_content)