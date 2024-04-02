import cv2


def get_activation_area(frame, face_coords):
    activation_area = (
        face_coords[0] - 400,
        face_coords[1] - 120,
        face_coords[0] + 400,
        face_coords[1] + 120,
    )

    cv2.rectangle(
        frame,
        (activation_area[0], activation_area[1]),
        (activation_area[2], activation_area[3]),
        (255, 0, 0),
        2,
    )

    return activation_area
