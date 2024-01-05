import os
import torch
import numpy as np
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration


# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:0.1' # Adjust the number based on trial and error

def Transcribe(audio_path, rttm_path, model_id, output_path, language):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    # model_id = "openai/whisper-large-v3"
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
    model.to(device)
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    # Set the output file paths
    output_file = output_path
    output_file_simple = output_path

    # Read the RTTM file and extract speaker timestamps
    with open(rttm_path, 'r') as rttm_file:
        speaker_data = [line.strip().split() for line in rttm_file]

    # Initialize variables for speaker tracking
    transcriptions = []

    print('Starting transcription')

    total_iterations = len(speaker_data)

    for i, line in enumerate(speaker_data):
        # Extract relevant data from the RTTM line
        _, audio_file, _, start_time, duration, _, _, speaker, _, _ = line
        start_time = float(start_time)
        duration = float(duration)

        # Calculate the percentage of iterations completed
        progress_percentage = (i + 1) / total_iterations * 100

        # Print the progress percentage
        print(f"Transcription progress: {progress_percentage:.2f}%")

        # Load the audio segment and specify the sampling rate (16,000 Hz in this case)
        audio, sampling_rate = librosa.load(audio_path, sr=16000, offset=start_time, duration=duration)

        # Process audio segment and generate transcription
        # transcription = pipe(audio, generate_kwargs = {"language":"<|nl|>"})
        transcription = pipe(audio)
        text = transcription["text"]

        # Append the results to the transcription list
        transcriptions.append((start_time, speaker, text))

    # Save the results to an output file
    with open(output_file, 'w', encoding = "utf-8") as f1, open(output_file_simple, 'w', encoding = "utf-8") as f2:
        for start_time, speaker, transcription in transcriptions:
            timestamp_line = f"Timestamp: {start_time:.2f}s, Speaker: {speaker}, Transcription: {transcription}\n"
            simple_line = f"{speaker}: {transcription}\n"
            
            # Write data to files (uncomment if needed)
            f1.write(timestamp_line)
            # f2.write(simple_line)

    print('Transcription done, file written to: ', output_path)

def merge_transcriptions(input_file, output_file):
    merged_output = []
    current_speaker = None
    current_transcription = ""
    current_timestamp = ""

    with open(input_file, 'r') as file:
        for line in file:
            parts = line.strip().split(", ", 2)  # Split only into three parts
            if len(parts) < 3:
                continue  # Skip invalid lines

            timestamp, speaker, transcription = parts

            if speaker == current_speaker:
                # Merge transcriptions, removing extra spaces
                current_transcription += " " + transcription.replace("Transcription: ", "").strip()
            else:
                # Output the previous segment if it exists
                if current_speaker is not None:
                    merged_output.append(f"{current_timestamp}, {current_speaker}, Transcription: {current_transcription.strip()}")

                # Start a new segment
                current_speaker = speaker
                current_transcription = transcription.replace("Transcription: ", "").strip()
                current_timestamp = timestamp

        # Add the last segment
        if current_speaker is not None:
            merged_output.append(f"{current_timestamp}, {current_speaker}, Transcription: {current_transcription.strip()}")

    # Save to output file
    with open(output_file, 'w') as file:
        for line in merged_output:
            file.write(line + "\n")

# Specify the file paths
# input_file = "outputtime.txt"
# output_file = "merged_output.txt"

# Read, merge, and save transcriptions
# merge_transcriptions(input_file, output_file)
# print(f"Merged transcriptions saved to {output_file}")


# Uncomment the following line to call the Transcribe function
# audio_path = '/home/evert/Desktop/audio/Ludenbos.mp3'
# rttm_path = "/home/evert/Desktop/audio/output.rttm"
# output_path = "/home/evert/Desktop/audio/outputSep.txt"
# model = "openai/whisper-large-v3"
# language = "dutch"
# Transcribe(audio_path, rttm_path, model, output_path, language)
