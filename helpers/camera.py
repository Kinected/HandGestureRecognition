import time

import cv2


def close_camera(capture):
    capture.release()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def read_frame(capture, framerate=None):
    ret, frame = capture.read()

    if framerate is not None:
        time.sleep(1 / framerate)

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        close_camera(capture)
        return None

    return frame


def get_close_event():
    return cv2.waitKey(1) & 0xFF == ord("q")


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
