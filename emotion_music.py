#Emotion based music recommendation system
import os

import cv2
import pandas as pd  # type: ignore
import pygame  # type: ignore
from deepface import DeepFace  # type: ignore

print("Current Folder:", os.getcwd())
print("Music Folder Exists:", os.path.exists("music"))
print("Angry MP3 Exists:", os.path.exists("music/angry.mp3"))
# Load songs dataset
songs = pd.read_csv("songs.csv")

# Initialize pygame mixer
pygame.mixer.init()


# Emotion -> MP3 mapping
music_files = {
    "happy": "music/happy.mp3",
    "sad": "music/sad.mp3",
    "angry": "music/angry.mp3",
    "neutral": "music/neutral.mp3",
    "fear": "music/fear.mp3",
    "surprise": "music/surprise.mp3"
}

# Open webcam
camera = cv2.VideoCapture(0)

print("Press SPACE to capture emotion")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Camera not detected!")
        break

    cv2.imshow("Emotion Detection", frame)

    key = cv2.waitKey(1)

    if key % 256 == 32:   # SPACE key
        break

# Detect emotion using DeepFace
try:
    result = DeepFace.analyze(
        frame,
        actions=['emotion'],
        enforce_detection=False
    )

    emotion = result[0]['dominant_emotion']

    print("\nDetected Emotion:", emotion)

    # Recommend songs
    recommendations = songs[
        songs["Emotion"].str.lower() == emotion.lower()
    ]

    print("\nRecommended Songs:")

    for song in recommendations["Song"]:
        print("🎵", song)

    # Play music
    if emotion in music_files:

        print("\nPlaying music...")

        pygame.mixer.music.load(
            music_files[emotion]
        )

        pygame.mixer.music.play()

        input(
            "\nPress Enter to stop music..."
        )

        pygame.mixer.music.stop()

    else:
        print("No music file found for emotion.")

except Exception as e:  # noqa: BLE001
    print("Error:", e)

camera.release()
cv2.destroyAllWindows()

