import cv2
import dlib
import numpy as np
import os
from tensorflow.keras.models import load_model

# Set working directory to where your files are
os.chdir(r"C:\Users\user\Desktop\AI_Image")

# Load the emotion recognition model (compile=False avoids optimizer errors)
emotion_model = load_model("emotion_model.h5", compile=False)

# Emotion labels and unhappy emotions list
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
unhappy_emotions = ['angry', 'disgust', 'fear', 'sad']

# Load Dlib face detector and shape predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the video with full path for reliability
video_path = r"C:\Users\user\Desktop\AI_Image\Love_Is_The_Answer.mp4"
video = cv2.VideoCapture(video_path)
if not video.isOpened():
    print("[ERROR] Could not open video file.")
    exit(1)

video_fps = video.get(cv2.CAP_PROP_FPS)
if video_fps == 0 or video_fps is None:
    print("[WARNING] Video FPS not detected. Using fallback 25 FPS.")
    video_fps = 25
video_delay = int(1000 / video_fps)

# Open webcam
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("[ERROR] Could not open webcam.")
    exit(1)

paused = False
last_video_frame = None  # To hold last video frame when paused

print("[INFO] Press 'q' to quit.")

while True:
    # Step 1: Read webcam frame
    ret_webcam, webcam_frame = webcam.read()
    if not ret_webcam:
        print("[ERROR] Cannot read webcam frame.")
        break

    gray = cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    emotion = "neutral"

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_roi = gray[y:y+h, x:x+w]

        try:
            # Resize face ROI to 64x64 for model input
            face_roi = cv2.resize(face_roi, (64, 64))
        except Exception as e:
            print(f"[WARNING] Resize failed: {e}")
            continue

        # Normalize pixel values and add batch & channel dims
        face_roi = face_roi.astype("float32") / 255.0
        face_roi = np.expand_dims(face_roi, axis=-1)  # Add channel dimension
        face_roi = np.expand_dims(face_roi, axis=0)   # Add batch dimension

        preds = emotion_model.predict(face_roi, verbose=0)[0]
        emotion = emotion_labels[np.argmax(preds)]
        break  # Use only the first detected face

    # Show emotion on webcam frame
    color = (0, 255, 0) if emotion == 'happy' else (0, 0, 255)
    cv2.putText(webcam_frame, f"Emotion: {emotion}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Step 2: Control video playback based on emotion
    if emotion in unhappy_emotions:
        paused = True
    elif emotion == "happy":
        paused = False

    if not paused:
        ret_video, video_frame = video.read()
        if not ret_video:
            print("[INFO] End of video reached.")
            break
        last_video_frame = video_frame
    else:
        # Stay on last frame while paused
        current_pos = video.get(cv2.CAP_PROP_POS_FRAMES)
        video.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_pos - 1))

    # Resize frames for side-by-side display
    if last_video_frame is not None:
        video_resized = cv2.resize(last_video_frame, (640, 480))
    else:
        video_resized = np.zeros((480, 640, 3), dtype=np.uint8)

    webcam_resized = cv2.resize(webcam_frame, (640, 480))

    # Concatenate horizontally
    combined = np.hstack((webcam_resized, video_resized))

    cv2.imshow("Webcam (left) + Video (right)", combined)

    # Quit on 'q' key
    if cv2.waitKey(video_delay) & 0xFF == ord('q'):
        break

# Cleanup
webcam.release()
video.release()
cv2.destroyAllWindows()
