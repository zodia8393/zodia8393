#VIDEO MUSIC CHANGER

# List of required packages
required_packages = ['selenium==4.0.0', 'pytube', 'pydub', 'keras', 'tensorflow', 'mido']

# Check if packages are installed, and install them if not
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"{package} is not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import sys
import subprocess
import midiutil
import time
import os
from selenium import webdriver
from pytube import YouTube
from pydub import AudioSegment
import librosa
import numpy as np
import tensorflow as tf


# Set up Chrome WebDriver
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--headless")  # Run Chrome in headless mode
chrome_options.add_argument("--disable-gpu")  # Disable GPU acceleration
driver = webdriver.Chrome(options=chrome_options)

# Navigate to YouTube and search for a keyword
driver.get("https://www.youtube.com/")
search_box = driver.find_element_by_name("search_query")
search_box.send_keys("korean music")
search_box.submit()

# Scroll down to load more videos
scroll_pause_time = 1
last_height = driver.execute_script("return document.documentElement.scrollHeight")
while True:
    driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
    time.sleep(scroll_pause_time)
    new_height = driver.execute_script("return document.documentElement.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

# Get the URLs of the first 10 videos
video_links = []
video_elements = driver.find_elements_by_xpath("//a[@class='yt-simple-endpoint style-scope ytd-video-renderer']")
for video_element in video_elements[:10]:
    video_link = video_element.get_attribute("href")
    video_links.append(video_link)

# Download and convert each video to WAV format
for video_link in video_links:
    # Download the video using pytube
    video = YouTube(video_link)
    video_streams = video.streams.filter(only_audio=True)
    audio_stream = video_streams.first()
    audio_filename = f"{video.video_id}.mp4"
    audio_path = os.path.join(os.getcwd(), audio_filename)
    audio_stream.download(output_path=os.getcwd(), filename=audio_filename)

    # Convert the audio to WAV format using pydub
    audio = AudioSegment.from_file(audio_path, format="mp4")
    wav_path = os.path.join(os.getcwd(), f"{video.video_id}.wav")
    audio.export(wav_path, format="wav")

    # Remove the downloaded MP4 file
    os.remove(audio_path)

# Close the Chrome WebDriver
driver.quit()

# Load the DSN model
model = tf.keras.models.load_model("DSN_model.h5")

# Set up constants for pitch detection
sampling_rate = 16000
frame_length = 2048
hop_length = 512
n_mels = 128

# Loop through each WAV file and detect pitch
for filename in os.listdir(os.getcwd()):
    if filename.endswith(".wav"):
        # Load the WAV file using librosa
        wav_path = os.path.join(os.getcwd(), filename)
        wav, sr = librosa.load(wav_path, sr=sampling_rate)

        # Preprocess the audio for the DSN model
        stft = librosa.stft(wav, n_fft=frame_length, hop_length=hop_length)
        spectrogram = np.abs(stft)
        mel_spec = librosa.feature.melspectrogram(S=spectrogram, sr=sampling_rate, n_mels=n_mels)
        log_mel_spec = librosa.amplitude_to_db(mel_spec)

        # Resize the spectrogram to match the input shape of the DSN model
        resized_spec = np.resize(log_mel_spec, (1, n_mels, log_mel_spec.shape[1], 1))

        # Make a prediction using the DSN model
        prediction = model.predict(resized_spec)
        predicted_pitch = np.argmax(prediction)

        # Print the filename and predicted pitch
        print(f"File {filename}: predicted pitch is {predicted_pitch}")


# Define a mapping from pitch indices to note names
note_names = {
    0: "C",
    1: "C#",
    2: "D",
    3: "D#",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "G#",
    9: "A",
    10: "A#",
    11: "B"
}

# Define the C major scale
scale = ["C", "D", "E", "F", "G", "A", "B"]

# Loop through each pitch data file and convert to musical score
for filename in os.listdir(os.getcwd()):
    if filename.endswith(".txt"):
        # Load the pitch data from the file
        pitch_path = os.path.join(os.getcwd(), filename)
        with open(pitch_path, "r") as f:
            pitch_data = [int(line.strip()) for line in f.readlines()]

        # Convert the pitch data to note names and filter out invalid pitches
        note_data = [note_names[pitch] if pitch in note_names else None for pitch in pitch_data]
        note_data = [note for note in note_data if note is not None]

        # Transpose the note data to the C major scale
        first_note = note_data[0]
        note_data = [scale[(scale.index(note) - scale.index(first_note)) % len(scale)] for note in note_data]

        # Print the musical score for the file
        score = " ".join(note_data)
        print(f"File {filename}: {score}")
        

# Define a mapping from pitch indices to note names
note_names = {
    0: "C",
    1: "C#",
    2: "D",
    3: "D#",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "G#",
    9: "A",
    10: "A#",
    11: "B"
}

# Define the C major scale
scale = ["C", "D", "E", "F", "G", "A", "B"]

# Loop through each pitch data file and convert to musical score
for filename in os.listdir(os.getcwd()):
    if filename.endswith(".txt"):
        # Load the pitch data from the file
        pitch_path = os.path.join(os.getcwd(), filename)
        with open(pitch_path, "r") as f:
            pitch_data = [int(line.strip()) for line in f.readlines()]

        # Convert the pitch data to note names and filter out invalid pitches
        note_data = [note_names[pitch] if pitch in note_names else None for pitch in pitch_data]
        note_data = [note for note in note_data if note is not None]

        # Transpose the note data to the C major scale
        first_note = note_data[0]
        note_data = [scale[(scale.index(note) - scale.index(first_note)) % len(scale)] for note in note_data]

        # Create a MIDI file for the score
        midi_path = os.path.splitext(pitch_path)[0] + ".mid"
        midi_file = midiutil.MIDIFile(1)
        midi_file.addTrackName(0, "Track 1")
        midi_file.addTempo(0, 0, 120)

        # Add each note in the score to the MIDI file
        time = 0
        for note in note_data:
            note_value = midiutil.note_name_to_value(note)
            midi_file.addNote(0, 0, note_value, time, 1, velocity=100)
            time += 1

        # Write the MIDI file to disk
        with open(midi_path, "wb") as f:
            midi_file.writeFile(f)

