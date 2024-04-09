import time

import numpy as np
from tensorflow.keras.models import load_model

from helpers.computations import compute_distances_angles_from_wrist
from helpers.gesture_handler.activation_area import get_activation_area
from helpers.gesture_handler.click_handler import ClickHandler
from helpers.gesture_handler.coordinates import get_face_coordinates, get_hand_coordinates, \
    is_hand_in_area_of_activation
from helpers.gesture_handler.landmarks import get_landmarks
from helpers.gesture_handler.swipe_handler import SwipeHandler
from helpers.mediapipe import draw_box, draw_face_pointer, draw_hand_pointer
from helpers.predictions import get_label

MIN_GESTURE_CONFIDENCE = 0.5
LABELS = [
    "closed",
    "palm",
    "point_up",
    "rock",
    "victory",
    "victory_inverted",
]

LOSE_FOCUS_AFTER_SECONDS = 2


class GestureHandler:
    swipe_handler = None
    click_handler = None

    current_action = None

    current_gesture = None

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

    def __init__(self, frame_resolution: tuple[int, int], holistic_model, gesture_model_path,
                 swipe_sensitivity={"x": 0.25, "y": 0.25}):

        self.current_action = None
        self.frame_resolution = frame_resolution

        self.gesture_model = load_model(gesture_model_path, compile=False)
        self.holistic_model = holistic_model

        self.start_time = None

        self.swipe_handler = SwipeHandler(frame_resolution, swipe_sensitivity)
        self.click_handler = ClickHandler()

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

    def draw_pointers(self, frame, activation_area=None):
        for key, coords in self.coordinates.items():
            if key == "face":
                draw_face_pointer(frame, coords)
            else:
                is_listened = key == self.hand_listened
                activated = is_hand_in_area_of_activation(coords, activation_area)
                draw_hand_pointer(frame, coords, activated, is_listened)

    def get_listening_hand(self, frame):
        """
        Get the listening hand in activation area
        :param frame: OpenCV Frame
        :return:
        """

        # Compute the area of activation from nose coordinates
        activation_area = get_activation_area(frame, self.coordinates["face"])
        self.draw_pointers(frame, activation_area)

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

    def update_no_interaction_since(self):
        if self.current_gesture in ["closed", "palm"]:
            self.no_interaction_since = None
            return

        if not self.no_interaction_since:
            if not self.swipe_handler.coords_locked:
                self.no_interaction_since = time.time()

            if self.hand_listened and self.coordinates[self.hand_listened] == (0, 0):
                self.no_interaction_since = time.time()

            return

        time_without_interaction = time.time() - self.no_interaction_since

        if time_without_interaction > LOSE_FOCUS_AFTER_SECONDS:
            self.hand_listened = None
            self.no_interaction_since = None

            # To avoid having the last gesture remaining at the last swipe seen
            self.swipe_handler.locked_control_coords = [
                (0, 0),
                (0, 0)
            ]

    def listen(self, frame, hand: str):
        """
        Gets actions from user based on their gestures and swipes
        :param frame: OpenCV frame
        :param hand: the hand currently watched
        :return: action of the user
        """

        landmarks = self.landmarks[hand]
        gesture, accuracy = self.get_gesture(hand, landmarks)

        self.current_gesture = gesture

        draw_box(frame, gesture, accuracy, hand, landmarks)

        coords_locked = self.swipe_handler.handle_locking(gesture)
        self.update_no_interaction_since()

        self.swipe_handler.update_locked_coords(self.coordinates, hand)

        self.click_handler.handle_step(gesture)

        last_swipe = self.swipe_handler.current_swipe
        self.current_action = self.swipe_handler.get_current_swipe()

        is_clicking = self.click_handler.is_clicking()

        if is_clicking:
            return "click"

        if coords_locked:
            self.swipe_handler.draw(frame)
            return "hover_" + self.current_action

        # We return the last swipe to be able to catch validated swipes:
        # If the last gesture was "hover_up-left" and the user open his hand, the current gesture will be "none",
        # not the validated "up-left". We keep the last_swipe in a variable to keep it
        return last_swipe
