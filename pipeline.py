import cv2
import asyncio
import mediapipe as mp
import websockets
from websockets.exceptions import ConnectionClosed
from tensorflow.keras.models import load_model
from helpers.websockets.send_gesture import send_gesture
from helpers.gesture_handler.gesture_handler import GestureHandler
from helpers.computations import compute_hand_coordinates
from helpers.predictions import get_gesture
from helpers.mediapipe import get_landmarks, draw_landmarks, draw_box
from helpers.camera import read_frame, show_frame, get_close_event, close_camera, draw_hand_pointer, flip_frame, \
    frame_preprocessing

MIN_DETECTION_CONFIDENCE = 0.4
MIN_PRESENCE_CONFIDENCE = 0.4
NUM_HANDS = 1
MIN_TRACKING_CONFIDENCE = 0.3

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

labels = [
    "closed",
    "dislike",
    "like",
    "palm",
    "point_up",
    "rock",
    "victory",
    "victory_inverted",
]

BOX_MARGIN = 24

DEBUG = False
FRAMERATE = None
RESOLUTION = (1280, 720)  # None  # (833, 480)  # (1280, 720)
FLIP_CAMERA = False

MODEL_NAME = "4_feb_w_additional_datasets"

MODEL_PATH = f"./models/{MODEL_NAME}/{MODEL_NAME}.hdf5"
MIN_GESTURE_CONFIDENCE = 0.7
MP_MODEL_COMPLEXITY = 0

HAND_CONTROL = "right"
SWIPE_THRESHOLD = 0.2

uri = "ws://localhost:8765/"


async def main():
    gesture_handler = GestureHandler(frame_resolution=RESOLUTION, swipe_threshold=SWIPE_THRESHOLD)

    capture = cv2.VideoCapture(1)

    model = load_model(MODEL_PATH, compile=False)

    hand = [0, 1] if HAND_CONTROL == "right" else [1, 0]

    with mp_holistic.Holistic(
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            model_complexity=MP_MODEL_COMPLEXITY,
    ) as holistic:
        while True:
            try:
                async with websockets.connect(uri) as websocket:
                    while True:
                        frame = read_frame(capture, FRAMERATE)
                        frame = frame_preprocessing(frame, RESOLUTION, FLIP_CAMERA)

                        landmarks = get_landmarks(frame, holistic)

                        hand_landmark = landmarks[HAND_CONTROL]

                        current_swipe = None

                        if hand_landmark:
                            if DEBUG: frame = draw_landmarks(frame, hand_landmark, mp_holistic, mp_drawing)

                            # Get hand coordinates (not normalized)
                            hand_coordinates = compute_hand_coordinates(frame, hand_landmark)

                            # Draw hand pointer
                            frame = draw_hand_pointer(frame, hand_coordinates)

                            # Get current gesture
                            gesture, accuracy = get_gesture(
                                model, labels, MIN_GESTURE_CONFIDENCE, hand, hand_landmark
                            )

                            # Draw box around hand
                            frame = draw_box(frame, gesture, accuracy, hand, hand_landmark)

                            # Handle gesture logic
                            current_swipe = gesture_handler.listen(frame, hand_coordinates, gesture)

                            if current_swipe:
                                print(current_swipe)
                                payload = {
                                    "gesture": gesture,
                                    "accuracy": float(accuracy),
                                    "hand": hand,
                                    "swipe": current_swipe,
                                }
                                await asyncio.create_task(send_gesture(websocket, payload))

                        frame = flip_frame(frame)

                        # Display swipe on frame
                        gesture_handler.draw_swipe(frame, current_swipe)
                        show_frame(frame, "hand gesture recognition")

                        if get_close_event():
                            break

            except ConnectionClosed:
                print("Connection lost. Reconnecting...")
                continue

            finally:
                close_camera(capture)


asyncio.run(main())
