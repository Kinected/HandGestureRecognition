def get_hand_coordinates(frame_resolution: tuple[int, int], landmarks) -> tuple[int, int]:
    """
    Compute the coordinates of the hand based on the landmarks
    :param frame_resolution: the frame resolution
    :param landmarks: hand landmarks from mediapipe
    :return: coordinates [x, y] of hand
    """

    landmark = landmarks.landmark[9]  # Index finger tip
    coordinates = (int(landmark.x * frame_resolution[0]), int(landmark.y * frame_resolution[1]))

    return coordinates


def get_face_coordinates(frame_resolution: tuple[int, int], landmarks) -> tuple[int, int]:
    """
    Get the coordinates of the face based on the landmarks
    :param frame_resolution: the frame resolution
    :param landmarks: face landmarks from mediapipe
    :return: coordinates [x, y] of face
    """
    landmark = landmarks.landmark[4]  # Nose
    coordinates = (int(landmark.x * frame_resolution[0]), int(landmark.y * frame_resolution[1]))

    return coordinates


def is_hand_in_area_of_activation(hand_coordinates, activation_area):
    """
    Check if the hand is in the area of activation
    :param hand_coordinates: coordinates of the hand
    :param activation_area: area of activation
    :return: boolean
    """

    x, y = hand_coordinates

    return activation_area[0] < x < activation_area[2] and activation_area[1] < y < activation_area[3]
