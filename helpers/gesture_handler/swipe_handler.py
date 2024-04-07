import cv2

# Swipe directions depending on direction
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


class SwipeHandler:
    current_swipe = None

    delta_thresholds = {
        "x": 0,
        "y": 0,
    }
    deltas = {
        "x": 0,
        "y": 0

    }
    coords_locked = False
    locked_control_coords = [
        (0, 0),
        (0, 0),
    ]

    def __init__(self, resolution=(0, 0), sensitivity: dict[str, float] = {"x": 0.25, "y": 0.25}):

        self.delta_thresholds = {
            "x": sensitivity["x"] * resolution[0],
            "y": sensitivity["y"] * resolution[1],
        }

    def handle_locking(self, current_gesture: str):
        """
        Function that locks hand position to calculate swipes
        :param current_gesture: the gesture currently being done
        :return: the new locking state
        """
        if current_gesture == "closed" and not self.coords_locked:
            self.coords_locked = True

        if current_gesture == "palm" and self.coords_locked:
            self.coords_locked = False
            self.locked_control_coords = [
                (0, 0),
                (0, 0),
            ]

        return self.coords_locked

    def update_locked_coords(self, coordinates: dict[str, tuple[int, int]], hand_listened: str):
        if not hand_listened or coordinates[hand_listened] == (0, 0):
            return

        if self.coords_locked:
            self.locked_control_coords[1] = coordinates[hand_listened]
            return

        self.locked_control_coords[0] = coordinates[hand_listened]

    def draw(self, frame):
        """
        Draw the current swipe
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

    def get_current_swipe(self):

        self.deltas = {
            "x": 0,
            "y": 0
        }

        # If locked coords are not set, return "none"
        if self.locked_control_coords[0] == (0, 0) or self.locked_control_coords[1] == (0, 0):
            return "none"

        # Update deltas

        deltaX = self.locked_control_coords[1][0] - self.locked_control_coords[0][0]
        deltaY = self.locked_control_coords[1][1] - self.locked_control_coords[0][1]

        self.deltas = {
            "x": deltaX,
            "y": deltaY
        }

        vertical_swipe_direction = 0
        horizontal_swipe_direction = 0

        if deltaX < -self.delta_thresholds["x"]:
            horizontal_swipe_direction = 1

        elif deltaX > self.delta_thresholds["x"]:
            horizontal_swipe_direction = -1

        if deltaY < -self.delta_thresholds["y"]:
            vertical_swipe_direction = 1

        elif deltaY > self.delta_thresholds["y"]:
            vertical_swipe_direction = -1

        return swipe_directions.get((vertical_swipe_direction, horizontal_swipe_direction), "none")
