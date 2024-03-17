import json
import os

def format_timestamp(seconds_float):
    total_seconds = int(float(seconds_float))
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"

def create_html_chat_message(timestamp, speaker, transcription, speakers=None, speakers_name=None):
    if speakers_name is None or speaker not in speakers_name:
        speaker_name = speaker  # Use the speaker ID as the name if no mapping is provided
    else:
        speaker_name = speakers_name[speaker]

    max_different_chatboxes = 4
    chatbox_kind = int(speaker[-1]) % max_different_chatboxes

    # speaker_class = "user1" if speaker == "SPEAKER_01" else "user2"  # Adjust as needed
    speaker_class = ''.join(['user',str(chatbox_kind)])
    return f'<div class="chat-message {speaker_class}">\n' \
           f'    <span class="user-name">{speaker_name}:</span>\n' \
           f'    <p>{transcription}</p>\n' \
           f'    <span class="timestamp">{timestamp}</span>\n' \
           f'</div>\n'


def update_speaker_names_in_html(html_file_path, name_mapping):
    """
    Update speaker names in the HTML file based on the provided name mapping.

    :param html_file_path: Path to the HTML file.
    :param name_mapping: A dictionary where keys are old names and values are new names.
    """
    with open(html_file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    for old_name, new_name in name_mapping.items():
        content = content.replace(old_name, new_name)

    with open(html_file_path, 'w', encoding='utf-8') as file:
        file.write(content)

def process_transcription_file(file_path):#, speakers=None):
    chat_messages = ""
    with open(file_path, 'r', encoding='utf-8') as file:
        transcriptions = json.load(file)

    for entry in transcriptions:
        # timestamp = format_timestamp(entry['start_time'])
        timestamp = format_timestamp(entry['timestamp'])
        chat_messages += create_html_chat_message(timestamp, entry['speaker'], entry['transcription'])# , speakers)

    html_output = f"""<!DOCTYPE html>
<html>
<head>
    <style>
    body {{
        font-family: Arial, sans-serif;
        margin: 0;
        padding: 0;
        background-color: #f4f4f4;
    }}

    .chat-container {{
        max-width: 800px;
        width: 100%;
        margin: 20px auto;
        border: 1px solid #ddd;
        padding: 10px;
        background-color: #f9f9f9;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }}

    .chat-title {{
        text-align: center;
        font-size: 24px;
        color: #333;
        margin-bottom: 10px;
    }}

    .chat-subtitle {{
        text-align: center;
        font-size: 18px;
        color: #555;
        margin-bottom: 20px;
    }}

    .chat-titlenote {{
        text-align: center;
        font-size: 15px;
        color: #666;
        margin-bottom: 20px;
    }}

    .chat-message {{
        border-radius: 5px;
        margin-bottom: 10px;
        padding: 10px;
        position: relative;
    }}

    .user0 {{
        background-color: #99c1de;
        text-align: left;
    }}

    .user1 {{
        background-color: #eddcd2;
        text-align: left;
    }}

    .user2 {{
        background-color: #d6e2e9;
        text-align: left;
    }}

    .user3 {{
        background-color: #fde2e4;
        text-align: left;
    }}

    .user4 {{
        background-color: #dbe7e4;
        text-align: left;
    }}

    .user5 {{
        background-color: #fad2e1;
        text-align: left;
    }}

    .user6 {{
        background-color: #fff0f6;
        text-align: left;
    }}

    .user7 {{
        background-color: #fff0f6;
        text-align: left;
    }}

    .user8 {{
        background-color: #fff0f6;
        text-align: left;
    }}
    .user-name {{
        font-weight: bold;
        display: block;
        margin-bottom: 2px;
    }}

    .chat-message p {{
        margin: 0;
    }}

    .timestamp {{
        font-size: 12px;
        color: #666;
        position: absolute;
        bottom: 5px;
        right: 10px;
    }}
    </style>
</head>
<body>
    <div class="chat-container">
        <h1 class="chat-title">{os.path.splitext(os.path.basename(file_path))[0]}</h1>
        <h2 class="chat-subtitle">This is an AI-powered transcription of a conversation.</h2>
        {chat_messages}
        <h3 class="chat-titlenote">his transcription has been by Evert van Noort. More information can be found on <a href="http://www.evertvannoort.com/AI">evertvannoort.com/AI</a></h3>
    </div>
</body>
</html>"""
    return html_output

# Example usage
# speakers = {"SPEAKER_01": "Alice", "SPEAKER_02": "Bob"}  # Adjust speaker names as necessary
# output_html = process_transcription_file('path/to/your/jsonfile.json', speakers)

# Write the HTML output to a file
# with open('output.html', 'w', encoding='utf-8') as f:
    # f.write(output_html)