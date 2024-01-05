import os
import torch
import librosa
import json
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Function to transcribe audio to text
def Transcribe(audio_path, rttm_path, model_id, output_path, language):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
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

    with open(rttm_path, 'r') as rttm_file:
        speaker_data = [line.strip().split() for line in rttm_file]

    transcriptions = []

    print('Starting transcription')

    total_iterations = len(speaker_data)

    for i, line in enumerate(speaker_data):
        _, audio_file, _, start_time, duration, _, _, speaker, _, _ = line
        start_time = float(start_time)
        duration = float(duration)

        # Calculate the percentage of iterations completed
        progress_percentage = (i + 1) / total_iterations * 100

        # Print the progress percentage
        print(f"Transcription progress: {progress_percentage:.2f}%")

        audio, sampling_rate = librosa.load(audio_path, sr=16000, offset=start_time, duration=duration)

        transcription = pipe(audio)
        text = transcription["text"]

        transcriptions.append({
            "start_time": start_time,
            "speaker": speaker,
            "transcription": text
        })

    with open(output_path, 'w', encoding="utf-8") as f:
        json.dump(transcriptions, f, ensure_ascii=False, indent=4)

    print('Transcription done, file written to: ', output_path)

# Function to merge transcriptions
def merge_transcriptions(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        transcriptions = json.load(file)

    merged_output = []
    current_speaker = None
    current_transcription = ""
    current_timestamp = ""

    for entry in transcriptions:
        speaker = entry['speaker']
        transcription = entry['transcription']
        timestamp = f"{entry['start_time']:.2f}"

        if speaker == current_speaker:
            current_transcription += " " + transcription.strip()
        else:
            if current_speaker is not None:
                merged_output.append({
                    "timestamp": current_timestamp,
                    "speaker": current_speaker,
                    "transcription": current_transcription.strip()
                })

            current_speaker = speaker
            current_transcription = transcription.strip()
            current_timestamp = timestamp

    if current_speaker is not None:
        merged_output.append({
            "timestamp": current_timestamp,
            "speaker": current_speaker,
            "transcription": current_transcription.strip()
        })

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(merged_output, file, ensure_ascii=False, indent=4)

# Example usage
# audio_path = '/path/to/audio.mp3'
# rttm_path = '/path/to/rttm.txt'
# output_path = '/path/to/output.json'
# model_id = "openai/whisper-large-v3"
# language = "dutch"
# Transcribe(audio_path, rttm_path, model_id, output_path, language)

# input_file = '/path/to/output.json'
# output_file = '/path/to/merged_output.json'
# merge_transcriptions(input_file, output_file)
