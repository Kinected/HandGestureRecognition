import cv2
import asyncio
import mediapipe as mp
import websockets
from websockets.exceptions import ConnectionClosed
from helpers.websockets.send_gesture import send_gesture
from helpers.gesture_handler.gesture_handler import GestureHandler
from helpers.predictions import get_gesture
from helpers.mediapipe import draw_hand_landmarks, draw_face_landmarks, draw_box
from helpers.camera import read_frame, show_frame, get_close_event, close_camera, flip_frame, \
    frame_preprocessing

MIN_DETECTION_CONFIDENCE = 0.4
MIN_PRESENCE_CONFIDENCE = 0.4
NUM_HANDS = 1
MIN_TRACKING_CONFIDENCE = 0.3

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

BOX_MARGIN = 24

DEBUG = False
FRAMERATE = None
RESOLUTION = (833, 480)  # (1280, 720)
FLIP_CAMERA = False

MODEL_NAME = "4_feb_w_additional_datasets"
MODEL_PATH = f"./models/{MODEL_NAME}/{MODEL_NAME}.hdf5"

MP_MODEL_COMPLEXITY = 0

SWIPE_THRESHOLD = 0.2

# hand = [0, 1] if HAND_CONTROL == "right_hand" else [1, 0]

uri = "ws://localhost:8000/ws/swipes"

capture = cv2.VideoCapture(1)


def handle_frame():
    frame = read_frame(capture, FRAMERATE)
    frame = frame_preprocessing(frame, RESOLUTION, FLIP_CAMERA)

    return frame


async def main():
    with mp_holistic.Holistic(
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            model_complexity=MP_MODEL_COMPLEXITY,
    ) as holistic:

        gesture_handler = GestureHandler(frame_resolution=RESOLUTION,
                                         holistic_model=holistic,
                                         gesture_model_path=MODEL_PATH,
                                         swipe_threshold=SWIPE_THRESHOLD)

        while True:
            try:
                async with websockets.connect(uri) as websocket:
                    while True:
                        frame = handle_frame()

                        gesture_handler.handle_frame(frame)
                        listening_hand = gesture_handler.get_listening_hand(frame)

                        if listening_hand:
                            swipe = gesture_handler.listen(frame, listening_hand)

                            payload = {
                                "hand": listening_hand,
                                "coordinates": gesture_handler.coordinates,
                                "gesture": gesture_handler.current_gesture,
                                "swipe": swipe,
                            }

                            await send_gesture(websocket, payload)

                        frame = flip_frame(frame)
                        # Display swipe on frame
                        show_frame(frame, "hand gesture recognition")

                        if get_close_event():
                            break

            except ConnectionClosed:
                print("Connection lost. Reconnecting...")
                continue

            finally:
                close_camera(capture)


asyncio.run(main())
