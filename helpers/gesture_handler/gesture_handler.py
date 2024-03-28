import time

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from helpers.computations import compute_distances_angles_from_wrist
from helpers.gesture_handler.coordinates import get_hand_coordinates, get_face_coordinates, \
    is_hand_in_area_of_activation
from helpers.mediapipe import draw_face_pointer, draw_hand_pointer, draw_box
from helpers.predictions import get_label

MIN_GESTURE_CONFIDENCE = 0.6
LABELS = [
    "closed",
    "dislike",
    "like",
    "palm",
    "point_up",
    "rock",
    "victory",
    "victory_inverted",
]

LOSE_FOCUS_AFTER = 5


def get_landmarks(frame, holistics):
    frame.flags.writeable = False
    results = holistics.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame.flags.writeable = True

    return {
        "face": results.face_landmarks or None,
        "left_hand": results.left_hand_landmarks or None,
        "right_hand": results.right_hand_landmarks or None,
    }


def get_activation_area(frame, face_coords):
    activation_area = (
        face_coords[0] - 320,
        face_coords[1] - 80,
        face_coords[0] + 320,
        face_coords[1] + 80,
    )

    cv2.rectangle(
        frame,
        (activation_area[0], activation_area[1]),
        (activation_area[2], activation_area[3]),
        (255, 0, 0),
        2,
    )

    return activation_area


class GestureHandler:
    current_swipe = None
    last_swipe = None
    current_gesture = None
    last_gesture = None

    is_listening_for_swipe = False
    hand_listened = None
    hand_pending = {
        "left_hand": False,
        "right_hand": False,
    }

    pending_from = {
        "left_hand": None,
        "right_hand": None,
    }

    no_interaction_since = None

    landmarks = {
        "left_hand": None,
        "right_hand": None,
        "face": None,
    }

    coordinates = {
        "left_hand": (0, 0),
        "right_hand": (0, 0),
        "face": (0, 0),
    }
    coords_locked = False
    locked_control_coords = [
        (0, 0),
        (0, 0),
    ]

    def __init__(self, frame_resolution: tuple[int, int], holistic_model, gesture_model_path, swipe_threshold=0.25):

        self.frame_resolution = frame_resolution

        self.gesture_model = load_model(gesture_model_path, compile=False)
        self.holistic_model = holistic_model

        self.start_time = None

        self.delta_threshold = swipe_threshold * frame_resolution[1]

    def handle_frame(self, frame):
        """
        Handle the frame and get the landmarks
        :param frame: OpenCV frame
        :return:
        """
        self.landmarks = get_landmarks(frame, self.holistic_model)
        self.compute_coordinates()

        return

    def compute_coordinates(self):
        self.coordinates = {key: (0, 0) for key in ["left_hand", "right_hand", "face"]}

        for key in self.coordinates.keys():
            if self.landmarks.get(key):
                if key == "face":
                    self.coordinates[key] = get_face_coordinates(self.frame_resolution, self.landmarks[key])
                else:
                    self.coordinates[key] = get_hand_coordinates(self.frame_resolution, self.landmarks[key])

    def get_gesture(self, hand: str, landmarks):

        if not landmarks:
            return "no_gesture", 0

        hand = [1, 0] if hand == "left_hand" else [0, 1]

        landmarks_distances_and_angles = compute_distances_angles_from_wrist(
            landmarks.landmark
        )

        input_data = np.concatenate((hand, landmarks_distances_and_angles))
        input_data = input_data.reshape(1, -1)

        predictions = self.gesture_model.predict(input_data, verbose=0)

        accuracy = np.max(predictions)

        if accuracy < MIN_GESTURE_CONFIDENCE:
            return "no_gesture", 0

        gesture = get_label(LABELS, predictions[0])

        return gesture, accuracy

    def draw_pointers(self, frame):
        for key, coords in self.coordinates.items():
            if key == "face":
                draw_face_pointer(frame, coords)
            else:
                is_listened = key == self.hand_listened
                activated = is_hand_in_area_of_activation(coords, get_activation_area(frame, self.coordinates["face"]))
                draw_hand_pointer(frame, coords, activated, is_listened)

    def get_listening_hand(self, frame):

        self.draw_pointers(frame)

        # Compute the area of activation from nose coordinates
        activation_area = get_activation_area(frame, self.coordinates["face"])

        # check if each hand is in the area of activation
        in_activation_area = {
            "left_hand": is_hand_in_area_of_activation(
                self.coordinates["left_hand"], activation_area
            ),
            "right_hand": is_hand_in_area_of_activation(
                self.coordinates["right_hand"], activation_area
            ),
        }

        # if hand is in the area of activation, check for gesture
        for hand, is_pending in in_activation_area.items():

            if self.hand_listened == hand:
                continue

            if not is_pending:
                self.hand_pending[hand] = False
                self.pending_from[hand] = None
                continue

            landmarks = self.landmarks[hand]

            gesture = self.get_gesture(
                hand, landmarks
            )

            if gesture[0] == "palm":

                self.hand_pending[hand] = True

                if not self.pending_from[hand]:
                    self.pending_from[hand] = time.time()

                if time.time() - self.pending_from[hand] > 1:
                    self.hand_listened = hand
                    self.hand_pending[hand] = False
                    self.pending_from[hand] = None

        return self.hand_listened

    def get_swipe(self):

        if self.locked_control_coords[0] == (0, 0) or self.locked_control_coords[1] == (0, 0):
            return "none"

        deltaX = self.locked_control_coords[1][0] - self.locked_control_coords[0][0]
        deltaY = self.locked_control_coords[1][1] - self.locked_control_coords[0][1]

        vertical_swipe_direction = 0
        horizontal_swipe_direction = 0

        if deltaY < -self.delta_threshold:
            vertical_swipe_direction = 1

        elif deltaY > self.delta_threshold:
            vertical_swipe_direction = -1

        if deltaX < -self.delta_threshold:
            horizontal_swipe_direction = 1

        elif deltaX > self.delta_threshold:
            horizontal_swipe_direction = -1

        swipe_directions = {
            (1, 0): "up",
            (-1, 0): "down",
            (0, 1): "right",
            (0, -1): "left",
            (1, 1): "up-right",
            (1, -1): "up-left",
            (-1, 1): "down-right",
            (-1, -1): "down-left",
        }

        return swipe_directions.get((vertical_swipe_direction, horizontal_swipe_direction), "none")

    def update_gesture(self, gesture):
        self.last_gesture = self.current_gesture
        self.current_gesture = gesture

    def handle_locking(self):

        if self.current_gesture == "closed" and not self.coords_locked:
            print("Locking...")
            self.coords_locked = True

        if self.current_gesture == "palm" and self.coords_locked:
            print("Unlocking...")
            self.coords_locked = False
            self.locked_control_coords = [
                (0, 0),
                (0, 0),
            ]

    def update_locked_coords(self):

        if not self.hand_listened:
            return

        if self.coordinates[self.hand_listened] == (0, 0):
            return

        if self.coords_locked:
            self.locked_control_coords[1] = self.coordinates[self.hand_listened]
            return

        self.locked_control_coords[0] = self.coordinates[self.hand_listened]

    def draw_gesture(self, frame):
        """
        Draw the line between the two points
        :param frame:
        :return: frame with the line drawn
        """

        frame = cv2.putText(
            frame,
            self.current_swipe,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

        if self.coords_locked:
            frame = cv2.line(
                frame,
                self.locked_control_coords[0],
                self.locked_control_coords[1],
                (255, 0, 0),
                2,
            )

        return frame

    def listen(self, frame, hand: str):
        """
        Listen to the gesture and coordinates and handle the lock/unlock
        :param frame: OpenCV frame
        :param hand: the hand to listen to
        :return:
        """

        landmarks = get_landmarks(frame, self.holistic_model)[hand]

        gesture, accuracy = self.get_gesture(hand, landmarks)
        self.update_gesture(gesture)

        if not self.coords_locked and not self.no_interaction_since and self.current_gesture not in ["closed", "palm"]:
            self.no_interaction_since = time.time()

        if not self.coords_locked and not self.no_interaction_since and self.hand_listened and self.coordinates[
            self.hand_listened] == (0, 0):
            print("No interaction since")
            self.no_interaction_since = time.time()

        if self.no_interaction_since and time.time() - self.no_interaction_since > LOSE_FOCUS_AFTER:
            self.hand_listened = None
            self.no_interaction_since = None

        draw_box(frame, gesture, accuracy, hand, landmarks)

        self.handle_locking()
        self.update_locked_coords()

        self.last_swipe = self.current_swipe
        self.current_swipe = self.get_swipe()

        self.draw_gesture(frame)

        if self.coords_locked:
            return "hover_" + self.current_swipe

        return self.last_swipe
