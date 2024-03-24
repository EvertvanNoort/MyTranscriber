import os
import time
import torch
import librosa
import json
from rich.progress import Progress
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Function to transcribe audio to text
def Transcribe(audio_path, rttm_path, model_id, output_path):#, language=None):
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
        chunk_length_s=40,
        batch_size=16,
        return_timestamps=False,
        torch_dtype=torch_dtype,
        device=device,
    )

    with open(rttm_path, 'r') as rttm_file:
        speaker_data = [line.strip().split() for line in rttm_file]

    transcriptions = []
    outputtext = []

    print('Starting transcription')

    total_iterations = len(speaker_data)

    with Progress() as progress:
    # Adding a task to the progress bar
        task = progress.add_task("[green]Transcribing...", total=total_iterations)

        for i, line in enumerate(speaker_data):
            _, audio_file, _, start_time, duration, _, _, speaker, _, _ = line
            start_time = float(start_time)
            duration = float(duration)

            audio, sampling_rate = librosa.load(audio_path, sr=16000, offset=start_time, duration=duration)

            # transcription = pipe(audio, generate_kwargs={"language": "dutch", "task": "transcribe"})
            transcription = pipe(audio, generate_kwargs={"task": "transcribe"})
            text = transcription["text"]
            outputtext.append(text)
            transcriptions.append({
                "timestamp": start_time,
                "speaker": speaker,
                "transcription": text
            })
            progress.update(task, advance=1)

    with open(output_path + ".json", 'w', encoding="utf-8") as f:
        json.dump(transcriptions, f, ensure_ascii=False, indent=4) 

    with open(output_path + ".txt",'w', encoding="utf-8") as f:       
        # f.write(outputtext)
        f.write('\n'.join(outputtext))

# Function to merge transcriptions
def merge_transcriptions(input_file, output_file):
    with open(input_file + ".json", 'r', encoding='utf-8') as file:
        transcriptions = json.load(file)

    merged_output = []
    current_speaker = None
    current_transcription = ""
    current_timestamp = ""

    for entry in transcriptions:
        speaker = entry['speaker']
        transcription = entry['transcription']
        timestamp = f"{entry['timestamp']:.2f}"

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
    with open(output_file + "_elaborate.txt",'w', encoding="utf-8") as f: 
        f.write("".join([current_timestamp," - ", current_speaker, ": ", current_transcription]))

    with open(output_file + ".json", 'w', encoding='utf-8') as file:
        json.dump(merged_output, file, ensure_ascii=False, indent=4)