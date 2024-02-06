import torch
import torchaudio
from pyannote.audio import Pipeline
from updateRTTM import splitLongFragments
from updateRTTM import removeShortFragments
from pyannote.audio.pipelines.utils.hook import ProgressHook

def Diarization(audio_path, rttm_path, model, num_speakers=None):
    # Create a diarization pipeline from a pretrained model
    token = "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"
    pipeline = Pipeline.from_pretrained(model, use_auth_token=token)
    # Set the execution device (CUDA if available)
    pipeline.to(torch.device("cuda:0"))

    print('Diarization started')
    waveform, sample_rate = torchaudio.load(audio_path)

    # Conditional execution based on num_speakers
    if num_speakers is None:
        # Let the model determine the number of speakers
        with ProgressHook() as hook:
            diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, hook=hook)
    else:
        # Use the specified number of speakers
        with ProgressHook() as hook:
            diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, hook=hook, num_speakers=num_speakers)
             
    # Dump the diarization output to disk using RTTM format
    with open(rttm_path, "w") as rttm_file:
        diarization.write_rttm(rttm_file)

    print('Diarization done')


# Example usage:
    input_rttm_path = rttm_path
    output_rttm_path = rttm_path

    # max_duration = 30       # Maximum duration for each fragment
    min_duration = 0.7     # Minimum duration for each fragment

    # new_end = splitLongFragments(input_rttm_path, output_rttm_path, max_duration)

# run the function as often as necessary
# this is a weird way to check if it is done
    # while type(new_end) != type([]):
        # new_end = splitLongFragments(output_rttm_path, output_rttm_path, max_duration)
        # pass

    removeShortFragments(input_rttm_path, output_rttm_path, min_duration)

    print('Fragments fixed')

