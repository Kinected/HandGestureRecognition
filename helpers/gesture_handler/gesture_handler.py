import cv2


class GestureHandler:
    frame_resolution = (0, 0)
    delta_threshold = 0

    current_swipe = None
    current_gesture = None
    last_gesture = None
    is_locked = False

    locked_coordinates = [
        [0, 0],
        [0, 0],
    ]

    def __init__(self, frame_resolution: tuple[int], swipe_threshold=0.25):
        self.frame_resolution = frame_resolution
        self.delta_threshold = swipe_threshold * frame_resolution[1]

    def get_swipe(self):
        deltaX = self.locked_coordinates[1][0] - self.locked_coordinates[0][0]
        deltaY = self.locked_coordinates[1][1] - self.locked_coordinates[0][1]

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

        self.current_swipe = swipe_directions.get((vertical_swipe_direction, horizontal_swipe_direction), "none")

        return self.current_swipe

    def update_gesture(self, gesture):
        self.last_gesture = self.current_gesture
        self.current_gesture = gesture

    def handle_locking(self):
        if self.current_gesture == "closed" and not self.is_locked:
            print("Locking...")
            self.is_locked = True
            return None

        if self.current_gesture == "palm" and self.is_locked:
            print("Unlocking...")
            self.is_locked = False
            self.locked_coordinates = [
                [0, 0],
                [0, 0],
            ]
            return self.current_swipe

    def update_coords(self, coordinates):

        if self.is_locked:
            self.locked_coordinates[1] = coordinates
            return

        self.locked_coordinates[0] = coordinates

    def draw_gesture(self, frame):
        """
        Draw the line between the two points
        :param frame:
        :return:
        """

        cv2.line(
            frame,
            (self.locked_coordinates[0][0], self.locked_coordinates[0][1]),
            (self.locked_coordinates[1][0], self.locked_coordinates[1][1]),
            (255, 0, 0),
            2,
        )

    def draw_swipe(self, frame, swipe):
        """
        Draw the swipe direction on the frame
        :param frame: OpenCV frame
        :param swipe: swipe direction
        :return:
        """

        if self.is_locked:
            cv2.putText(
                frame,
                swipe,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

    def listen(self, frame, coordinates, gesture=None):
        """
        Listen to the gesture and coordinates and handle the lock/unlock
        :param frame: OpenCV frame
        :param coordinates: coordinates of the hand
        :param gesture: gesture detected
        :return:
        """

        self.update_gesture(gesture)

        swipe = self.handle_locking()
        self.update_coords(coordinates)

        if not swipe and self.is_locked:
            self.draw_gesture(frame)
            swipe = "hover_" + self.get_swipe()

        return swipe
