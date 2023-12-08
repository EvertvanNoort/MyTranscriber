import torch
import os
import numpy as np
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def Transcribe(audio_path, rttm_path, model, output_path, language):
    # Check for GPU availability and set the device
    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    # Set max_split_size_mb to 0.5 GB
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "0.01"

    # Initialize Whisper processor and model
    processor = WhisperProcessor.from_pretrained(model)
    model = WhisperForConditionalGeneration.from_pretrained(model).to(device)

    # Prepare decoder prompts
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
    model.config.forced_decoder_ids = None

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
        input_features = processor(audio, sampling_rate=sampling_rate, return_tensors="pt", max_new_tokens=4000).input_features
        predicted_ids = model.generate(input_features.to(device), forced_decoder_ids=forced_decoder_ids)
        # predicted_ids = model.generate(input_features.to(device))#, forced_decoder_ids=forced_decoder_ids)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        text = transcription[0]

        # Append the results to the transcription list
        transcriptions.append((start_time, speaker, text))

    # Save the results to an output file
    with open(output_file, 'w') as f1, open(output_file_simple, 'w') as f2:
        for start_time, speaker, transcription in transcriptions:
            timestamp_line = f"Timestamp: {start_time:.2f}s, Speaker: {speaker}, Transcription: {transcription}\n"
            simple_line = f"{speaker}: {transcription}\n"
            
            # Write data to files (uncomment if needed)
            # f1.write(timestamp_line)
            f2.write(simple_line)

    print('Transcription done, file written to: ', output_path)

# Uncomment the following line to call the Transcribe function
# Transcribe(audio_path, rttm_path, model, output_path, language)
