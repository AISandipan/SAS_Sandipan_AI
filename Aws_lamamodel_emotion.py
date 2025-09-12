

import cv2
import boto3
import numpy as np
import requests
import json
import os

# ========== AWS Configuration ==========
rekognition = boto3.client('rekognition', region_name='us-east-1')  # Adjust region

# ========== LLM Configuration ==========
LLM_API_URL = "http://localhost:8000/api"  # Replace with your actual LLaMA or LLM endpoint

# ========== Video & Webcam Setup ==========
video_path = r"C:\Users\user\Desktop\AI_Image\Love_Is_The_Answer.mp4"
video = cv2.VideoCapture(video_path)
webcam = cv2.VideoCapture(0)

if not video.isOpened() or not webcam.isOpened():
    print("[ERROR] Could not open video or webcam.")
    exit(1)

video_fps = video.get(cv2.CAP_PROP_FPS) or 25
video_delay = int(1000 / video_fps)

paused = False
last_video_frame = None

print("[INFO] Press 'q' to quit.")

# ========== Main Loop ==========
while True:
    ret_webcam, frame = webcam.read()
    if not ret_webcam:
        print("[ERROR] Cannot read webcam frame.")
        break

    # Convert webcam frame to JPEG for AWS Rekognition
    _, jpeg_data = cv2.imencode('.jpg', frame)
    response = rekognition.detect_faces(
        Image={'Bytes': jpeg_data.tobytes()},
        Attributes=['ALL']
    )

    # Default emotion
    emotion = "neutral"

    # Extract emotion from Rekognition response
    if response['FaceDetails']:
        emotions = response['FaceDetails'][0].get('Emotions', [])
        if emotions:
            # Take the most confident emotion
            emotion = max(emotions, key=lambda x: x['Confidence'])['Type'].lower()

    # ========== LLM Decision ==========
    llm_payload = {
        "emotion": emotion,
        "description": f"The person appears to be {emotion}. Should the video be paused?"
    }

    try:
        llm_response = requests.post(LLM_API_URL, json=llm_payload)
        llm_decision = llm_response.json().get("pause", False)
        paused = llm_decision
    except Exception as e:
        print(f"[WARNING] LLM call failed: {e}")
        paused = emotion in ['angry', 'disgusted', 'fear', 'sad']  # Fallback logic

    # Display emotion on webcam frame
    color = (0, 255, 0) if not paused else (0, 0, 255)
    cv2.putText(frame, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Video logic
    if not paused:
        ret_video, video_frame = video.read()
        if not ret_video:
            print("[INFO] End of video.")
            break
        last_video_frame = video_frame
    else:
        # Stay on last frame
        current_pos = video.get(cv2.CAP_PROP_POS_FRAMES)
        video.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_pos - 1))

    # Resize and combine
    if last_video_frame is not None:
        video_resized = cv2.resize(last_video_frame, (640, 480))
    else:
        video_resized = np.zeros((480, 640, 3), dtype=np.uint8)

    webcam_resized = cv2.resize(frame, (640, 480))
    combined = np.hstack((webcam_resized, video_resized))

    cv2.imshow("Webcam (left) + Video (right)", combined)

    if cv2.waitKey(video_delay) & 0xFF == ord('q'):
        break

# Cleanup
video.release()
webcam.release()
cv2.destroyAllWindows()
