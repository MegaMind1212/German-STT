import os
import re
import unicodedata
import logging
from flask import Flask, request, render_template, send_file
import torch
import torchaudio
from transformers import pipeline
from moviepy.editor import VideoFileClip
import io

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)

# Create upload and temp folders
UPLOAD_FOLDER = 'Uploads'
TEMP_FOLDER = 'Temp'
for folder in [UPLOAD_FOLDER, TEMP_FOLDER]:
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
            logging.info(f"Created folder: {folder}")
        except Exception as e:
            logging.error(f"Failed to create folder {folder}: {str(e)}")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMP_FOLDER'] = TEMP_FOLDER

# Load Whisper model (tiny for speed, switch to small for better accuracy)
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny",  # Change to "openai/whisper-small" for better accuracy
        device=device
    )
    logging.info("Whisper model loaded successfully")
except Exception as e:
    logging.error(f"Failed to load Whisper model: {str(e)}")

# Verbal punctuation mapping
punctuation_map = {
    "ausrufezeichen": "!",
    "fragezeichen": "?",
    "punkt": ".",
    "komma": ",",
    "semikolon": ";",
    "doppelpunkt": ":"
}

# Simple dictionary of common German nouns
german_nouns = {
    "buch": "Buch",
    "musik": "Musik",
    "haus": "Haus",
    "auto": "Auto",
    "baum": "Baum",
    "freund": "Freund",
    "schule": "Schule",
    "lehrer": "Lehrer",
    "kind": "Kind",
    "stadt": "Stadt"
}

def sanitize_filename(filename):
    """Remove special characters and emojis from filename."""
    filename = unicodedata.normalize('NFKD', filename).encode('ascii', 'ignore').decode('ascii')
    filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    base, ext = os.path.splitext(filename)
    if not ext.lower() in ('.mp3', '.mp4', '.wav'):
        ext = '.mp3'
    return base + ext

def capitalize_german_nouns(text):
    """Capitalize German nouns using a dictionary."""
    words = text.split()
    result = []
    for word in words:
        lower_word = word.lower()
        if lower_word in german_nouns:
            result.append(german_nouns[lower_word])
        else:
            result.append(word)
    return " ".join(result)

def handle_verbal_punctuation(text):
    """Replace verbal punctuation with symbols."""
    for verbal, symbol in punctuation_map.items():
        text = re.sub(r'\b' + verbal + r'\b', symbol, text, flags=re.IGNORECASE)
    return text

def is_valid_audio_file(file_path):
    """Check if the file exists, is non-empty, and has a valid audio header."""
    if not os.path.isfile(file_path):
        return False, "File does not exist"
    if os.path.getsize(file_path) == 0:
        return False, "File is empty"
    if file_path.lower().endswith(('.mp3', '.wav')):
        try:
            with open(file_path, 'rb') as f:
                header = f.read(4)
                if file_path.lower().endswith('.mp3') and not header.startswith(b'\xFF\xFB'):
                    return False, "Invalid MP3 header"
                if file_path.lower().endswith('.wav') and not header.startswith(b'RIFF'):
                    return False, "Invalid WAV header"
        except Exception as e:
            return False, f"Header check failed: {str(e)}"
    return True, ""

def extract_audio_from_mp4(mp4_path, temp_folder):
    """Extract audio from MP4 and save as WAV."""
    try:
        video = VideoFileClip(mp4_path)
        wav_path = os.path.join(temp_folder, "extracted_audio.wav")
        video.audio.write_audiofile(wav_path, codec='pcm_s16le', fps=16000)
        video.close()
        logging.info(f"Extracted audio from {mp4_path} to {wav_path}")
        return wav_path
    except Exception as e:
        raise RuntimeError(f"Failed to extract audio from MP4: {str(e)}")

def transcribe_audio(file_path, temp_folder):
    """Transcribe audio using Whisper."""
    logging.info(f"Attempting to transcribe: {file_path}")
    
    # Handle MP4 by extracting audio
    if file_path.lower().endswith('.mp4'):
        wav_path = extract_audio_from_mp4(file_path, temp_folder)
        file_path = wav_path
    
    # Validate file
    is_valid, error = is_valid_audio_file(file_path)
    if not is_valid:
        if file_path.lower().endswith('.mp4') or file_path.lower().endswith('.wav'):
            if os.path.exists(file_path):
                os.remove(file_path)
        raise ValueError(error)
    
    # Load audio with torchaudio
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        logging.info(f"Audio loaded: {file_path}, sample rate: {sample_rate}, duration: {waveform.shape[1] / sample_rate:.2f} seconds")
    except Exception as e:
        if file_path.lower().endswith('.mp4') or file_path.lower().endswith('.wav'):
            if os.path.exists(file_path):
                os.remove(file_path)
        raise RuntimeError(f"Failed to load audio: {str(e)}")
    
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Convert to numpy array for Whisper
    audio = waveform.squeeze().numpy()
    
    # Transcribe with timestamps for long audio
    try:
        result = pipe(audio, generate_kwargs={"language": "german", "return_timestamps": True})
        transcription = result["text"]
        logging.info(f"Transcription completed: {transcription[:100]}... (total length: {len(transcription)} characters)")
    except Exception as e:
        if file_path.lower().endswith('.mp4') or file_path.lower().endswith('.wav'):
            if os.path.exists(file_path):
                os.remove(file_path)
        raise RuntimeError(f"Transcription failed: {str(e)}")
    
    # Clean up WAV file if MP4
    if file_path.lower().endswith('.mp4') or file_path.lower().endswith('.wav'):
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Cleaned up: {file_path}")
    
    return transcription

def post_process_transcription(transcription):
    """Apply German-specific post-processing."""
    transcription = handle_verbal_punctuation(transcription)
    transcription = capitalize_german_nouns(transcription)
    return transcription

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>German Speech-to-Text</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #333; }
        .container { max-width: 600px; margin: auto; }
        .result { margin-top: 20px; padding: 10px; border: 1px solid #ccc; }
        .error { color: red; }
        .download { margin-top: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>German Speech-to-Text</h1>
        <p>Upload an MP3, MP4, or WAV file (MP3/MP4 preferred, WAV as fallback).</p>
        <form method="post" enctype="multipart/form-data" action="/transcribe">
            <input type="file" name="audio" accept=".mp3,.mp4,.wav" required>
            <input type="submit" value="Transcribe">
        </form>
        {% if transcription %}
        <div class="result">
            <h3>Transcription:</h3>
            <p>{{ transcription }}</p>
            <div class="download">
                <a href="/download_transcription" download="transcription.txt">Download Transcription</a>
            </div>
        </div>
        {% endif %}
        {% if error %}
        <div class="error">
            <p>Error: {{ error }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return render_template('index.html', error="No audio file provided")
    file = request.files['audio']
    if file.filename == '':
        return render_template('index.html', error="No file selected")
    if file and file.filename.lower().endswith(('.mp3', '.mp4', '.wav')):
        safe_filename = sanitize_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        logging.info(f"Attempting to save file: {file_path}")
        try:
            # Save file directly
            with open(file_path, 'wb') as f:
                f.write(file.read())
            logging.info(f"File saved: {file_path}, size: {os.path.getsize(file_path)} bytes")
            # Verify file
            if not os.path.isfile(file_path):
                raise ValueError(f"Failed to save file to {file_path}")
            if os.path.getsize(file_path) == 0:
                raise ValueError(f"Saved file {file_path} is empty")
            # Transcribe
            transcription = transcribe_audio(file_path, app.config['TEMP_FOLDER'])
            transcription = post_process_transcription(transcription)
            # Store transcription for download
            app.config['TRANSCRIPTION'] = transcription
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"Cleaned up: {file_path}")
            logging.error(f"Error during processing: {str(e)}")
            return render_template('index.html', error=f"Error: {str(e)}")
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Cleaned up: {file_path}")
        return render_template('index.html', transcription=transcription)
    return render_template('index.html', error="Invalid file format, please upload an MP3, MP4, or WAV")

@app.route('/download_transcription')
def download_transcription():
    transcription = app.config.get('TRANSCRIPTION', '')
    if not transcription:
        return render_template('index.html', error="No transcription available for download")
    buffer = io.StringIO(transcription)
    buffer.seek(0)
    return send_file(
        io.BytesIO(buffer.getvalue().encode('utf-8')),
        mimetype='text/plain',
        as_attachment=True,
        download_name='transcription.txt'
    )

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    with open('templates/index.html', 'w') as f:
        f.write(HTML_TEMPLATE)
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logging.error(f"Failed to start Flask app: {str(e)}")