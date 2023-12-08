import re

def format_timestamp(seconds_float):
    total_seconds = int(float(seconds_float))
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"

def parse_line(line):
    # Using regex to match the pattern in the line
    match = re.match(r"Timestamp: (\d+\.\d+)s, Speaker: (\w+), Transcription: (.+)", line)
    if match:
        timestamp_seconds, speaker, transcription = match.groups()
        timestamp = format_timestamp(timestamp_seconds)
        return timestamp, speaker, transcription
    else:
        return None, None, None  # Return None values if the line doesn't match the expected pattern

def create_html_chat_message(timestamp, speaker, transcription, speakers):
    speaker_name = speakers.get(speaker, speaker)  # Get the name from the dictionary, default to the speaker ID if not found
    speaker_class = "user1" if speaker == "SPEAKER_01" else "user2"  # Adjust this as needed for more speakers
    return f'<div class="chat-message {speaker_class}">\n' \
           f'    <span class="user-name">{speaker_name}:</span>\n' \
           f'    <p>{transcription}</p>\n' \
           f'    <span class="timestamp">{timestamp}</span>\n' \
           f'</div>\n'

def process_transcription_file(file_path, speakers):
    chat_messages = ""
    with open(file_path, 'r') as file:
        for line in file:
            timestamp, speaker, transcription = parse_line(line)
            if timestamp is not None:
                chat_messages += create_html_chat_message(timestamp, speaker, transcription, speakers)

    # Embedding the CSS in the HTML file
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

    .user1 {{
        background-color: #e6f7ff;
        text-align: left;
    }}

    .user2 {{
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
        <h1 class="chat-title">Unleashing the Creative Power of Language Models</h1>
        <h2 class="chat-subtitle">This is an AI-powered transcription of a conversation. The conversation can be found via this <a href="https://www.youtube.com/watch?v=LAt7q-Qsfi8&list=PPSV">link</a>:</h2>
        {chat_messages}
        <h3 class="chat-titlenote">his transcription has been by Evert van Noort. More information can be found on <a href="http://www.evertvannoort.com/AI">evertvannoort.com/AI</a></h3>
    </div>
</body>
</html>"""
    return html_output
