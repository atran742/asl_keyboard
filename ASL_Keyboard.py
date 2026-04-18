import cv2
import numpy as np
import mediapipe as mp
import joblib

def draw_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape

    # Define connections 
    connections = [
        (0,1),(1,2),(2,3),(3,4),      # thumb
        (0,5),(5,6),(6,7),(7,8),      # index
        (5,9),(9,10),(10,11),(11,12), # middle
        (9,13),(13,14),(14,15),(15,16), # ring
        (13,17),(17,18),(18,19),(19,20), # pinky
        (0,17) # palm base
    ]

    # Draw points
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    # Draw lines
    for start, end in connections:
        x1, y1 = int(hand_landmarks[start].x * w), int(hand_landmarks[start].y * h)
        x2, y2 = int(hand_landmarks[end].x * w), int(hand_landmarks[end].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

# For Reproducability
model = joblib.load("asl_model.pkl")
le = joblib.load("label_encoder.pkl") 

# Text tracking
typed_text = ""
current_prediction = ""

# Movement detection (for J/Z)
hand_moving = False
prev_tip_pos = None
MOVEMENT_THRESHOLD = 0.01

# Hold logic
prediction_hold = ""
prediction_hold_timer = 0
HOLD_FRAMES = 10

#For new Mediapipe API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

detector = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

frame_timestamp = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Run detection
    result = detector.detect_for_video(mp_image, frame_timestamp)
    frame_timestamp += 1

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        draw_landmarks(frame, hand) 

        # Build feature vector
        landmarks = []
        for lm in hand:
            landmarks.extend([lm.x, lm.y, lm.z])
        landmarks = np.array(landmarks).reshape(1, -1)

        # Predict
        prediction = model.predict(landmarks)
        current_prediction = le.inverse_transform(prediction)[0]

        # Movement detection
        tip = hand[8]  # index finger tip

        if prev_tip_pos is not None:
            movement = abs(tip.x - prev_tip_pos[0]) + abs(tip.y - prev_tip_pos[1])
            hand_moving = movement > MOVEMENT_THRESHOLD
            #Z and J are just i and d position but just moving, this detects if the user is moving their hand while holding up z and j 
            if hand_moving:
                if current_prediction == "I":
                    current_prediction = "J"
                    prediction_hold = "J"
                    prediction_hold_timer = HOLD_FRAMES
                elif current_prediction == "D":
                    current_prediction = "Z"
                    prediction_hold = "Z"
                    prediction_hold_timer = HOLD_FRAMES

        prev_tip_pos = (tip.x, tip.y)

    else:
        prev_tip_pos = None
        hand_moving = False

    # Hold logic
    if prediction_hold_timer > 0:
        current_prediction = prediction_hold
        prediction_hold_timer -= 1

    # UI
    cv2.rectangle(frame, (0, 0), (640, 80), (0, 0, 0), -1)

    cv2.putText(frame,
                "Prediction: " + current_prediction,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2)

    cv2.putText(frame,
                "Typed: " + typed_text,
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2)

    cv2.imshow("ASL Typing", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 13:  #Press enter to save your letter
        if current_prediction == "DELETE":
            typed_text = typed_text[:-1]
        elif current_prediction == "SPACE":
            typed_text += " "
        else:
            typed_text += current_prediction

    if key == 27:  #Press ESQ to leave the screen 
        break

cap.release()
cv2.destroyAllWindows()
