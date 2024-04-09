import asyncio

import cv2
import mediapipe as mp
import websockets

from helpers.camera import close_camera, flip_frame, frame_preprocessing, get_close_event, read_frame, show_frame
from helpers.gesture_handler.gesture_handler import GestureHandler
from helpers.websockets.websockets import connect, send_gesture

MIN_DETECTION_CONFIDENCE = 0.4
MIN_PRESENCE_CONFIDENCE = 0.4
NUM_HANDS = 1
MIN_TRACKING_CONFIDENCE = 0.2

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

BOX_MARGIN = 24

DEBUG = False
FRAMERATE = 20

FLIP_CAMERA = False

MODEL_NAME = "March28"
MODEL_PATH = f"./models/{MODEL_NAME}/{MODEL_NAME}.keras"

MP_MODEL_COMPLEXITY = 0

SWIPE_SENSITIVITY = {"x": 0.1, "y": 0.1}

# hand = [0, 1] if HAND_CONTROL == "right_hand" else [1, 0]

WEBSOCKET_URI = "ws://localhost:8000/ws/swipes"

capture = cv2.VideoCapture(0)

# RESOLUTION = (capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

RESIZE_TO = (800, 1000)  # (800, 800)
RESOLUTION = (400, 500)  # (800, 800)  # (1280, 720)


def handle_frame():
    """
    Function that gets frame from camera and preprocess it (resize, rescale, flip)
    :return: The preprocessed frame
    """
    frame = read_frame(capture, FRAMERATE)
    frame = frame_preprocessing(frame, RESIZE_TO, RESOLUTION, FLIP_CAMERA)

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
                                         swipe_sensitivity=SWIPE_SENSITIVITY)

        while True:
            websocket = await connect(WEBSOCKET_URI)
            try:
                while True:
                    frame = handle_frame()

                    gesture_handler.handle_frame(frame)
                    listening_hand = gesture_handler.get_listening_hand(frame)

                    payload = {
                        "hand": None,
                        "coordinates": gesture_handler.coordinates,
                        "gesture": "no_gesture",
                        "deltas": {"x": 0, "y": 0},
                        "action": None,
                    }

                    if listening_hand:
                        action = gesture_handler.listen(frame, listening_hand)

                        payload = {
                            "hand": listening_hand,
                            "coordinates": gesture_handler.coordinates,
                            "gesture": gesture_handler.current_gesture,
                            "deltas": gesture_handler.swipe_handler.deltas,
                            "action": action,
                        }

                    await send_gesture(websocket, payload)
                    frame = flip_frame(frame)
                    show_frame(frame, "hand gesture recognition")

                    if get_close_event():
                        print("Stop")
                        break

            except websockets.ConnectionClosed:
                print("Connection lost. Reconnecting...")
                continue

    close_camera(capture)


asyncio.run(main())
