import cv2


def get_landmarks(frame, holistics):
    """
    Get the mediapipe landmarks from the frame
    :param frame:
    :param holistics:
    :return: landmarks dict
    """
    if frame is None:
        return {
            "face": None,
            "left_hand": None,
            "right_hand": None,
        }

    frame.flags.writeable = False
    results = holistics.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame.flags.writeable = True

    return {
        "face": results.face_landmarks or None,
        "left_hand": results.left_hand_landmarks or None,
        "right_hand": results.right_hand_landmarks or None,
    }
