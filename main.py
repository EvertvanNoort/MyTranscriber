import os
import zipfile
from speakerProcessor import Diarization
from transcriber import Transcribe, merge_transcriptions
from writeChat import process_transcription_file
from writeChat import update_speaker_names_in_html
from extractive_summary import get_important_sentences

diarization_model = "pyannote/speaker-diarization-3.0"
transcription_model = "openai/whisper-large-v2"
# transcription_model = "openai/whisper-large-v3"
# transcription_model = "openai/whisper-tiny"

# Function to process a single audio file
def process_audio_file(audio_path, num_speakers, HTML):
    # print("made it here")
    # input_file = 'transcription_input.txt'
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    rttm_path = f"/home/evert/Desktop/audio/{base_name}.rttm"
    output_path = f"/home/evert/Desktop/audio/{base_name}"
    html_path = f"/home/evert/Desktop/audio/{base_name}.html"
    summary_path = f"/home/evert/Desktop/audio/{base_name}_sum.txt"

    Diarization(audio_path, rttm_path, diarization_model, num_speakers)
    Transcribe(audio_path, rttm_path, transcription_model, output_path)#, language=None)
    merge_transcriptions(output_path, output_path)

    if (HTML==1):
        html_content = process_transcription_file(output_path + ".json")# , speakers)#, speaker_names)
        # Write the result to an HTML file
        with open(html_path, 'w') as file:
            file.write(html_content)
        print("HTML construction done, file written to:", html_path)
    # The names of the files to be zipped
    file_names = [output_path + ".json",output_path + ".txt",output_path + "_elaborate.txt",html_path]
    zip_name = output_path

    # Create a ZIP file
    with zipfile.ZipFile(zip_name + ".zip", 'w') as myzip:
        for file_name in file_names:
            # Add file to the ZIP file
            base_file_name = os.path.basename(file_name)
            # Add file to the ZIP file with only the base filename
            myzip.write(file_name, arcname=base_file_name)

        print(f'Created {zip_name} containing {file_names}')

    if (SUM==1):
        get_important_sentences(output_path + ".txt", summary_path, prob_threshold=0.8)

HTML = 1
SUM = 0

input_file = 'transcription_input.txt'
# Read the input file and process each audio file listed
with open(input_file, 'r') as file:
    for line in file:
        line = line.strip()
        if line=='HTML':
            HTML = 1
            continue
        if line =='SUM':
            SUM = 1
            continue

        if not line or line.startswith('#'):
            continue  # Skip empty lines and lines starting with '#'

        parts = line.split(', ')
        audio_path = parts[0]
        num_speakers = None if len(parts) == 1 else int(parts[1])
        process_audio_file(audio_path, num_speakers, HTML)

# Example usage
# html_file_path = '/home/evert/Desktop/audio/Altman.html'
# name_mapping = {
#     'SPEAKER_00': 'Sam Altman',
#     'SPEAKER_01': 'Elevate Host',
#     # Add more mappings as needed
# }
# update_speaker_names_in_html(html_file_path, name_mapping)
