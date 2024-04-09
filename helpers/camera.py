import time

import cv2


def close_camera(capture):
    capture.release()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def read_frame(capture, framerate=None):
    if framerate is not None:
        time.sleep(1 / framerate)

    ret, frame = capture.read()

    return frame


def get_close_event():
    return cv2.waitKey(1) & 0xFF == ord("q")


def get_key_pressed():
    return cv2.waitKey(1) & 0xFF


def show_frame(frame, window_name="hand_gesture"):
    cv2.imshow(window_name, frame)


def show_bindings(frame):
    cv2.putText(
        frame,
        "Press 'q' to quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "a : Palm | z : Closed | e : like | r : dislike",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return frame


def open_image(path):
    image = cv2.imread(path)
    return image


def show_image(image):
    cv2.imshow("Image", image)
    cv2.waitKey(0)


def frame_preprocessing(frame, resize_to=None, resolution=None, flip=False):
    if resize_to is not None:
        frame_height, frame_width = frame.shape[:2]

        offset_x = int((frame_width - resize_to[0]) / 2)
        offset_y = int((frame_height - resize_to[1]) / 2)

        frame = frame[offset_y:offset_y + resize_to[1], offset_x:offset_x + resize_to[0]]

    if resolution is not None:
        frame = cv2.resize(frame, resolution)

    if flip:
        frame = cv2.flip(frame, 1)

    return frame


def flip_frame(frame):
    return cv2.flip(frame, 1)


def close_image():
    cv2.waitKey(2)
    cv2.destroyAllWindows()
    cv2.waitKey(2)
