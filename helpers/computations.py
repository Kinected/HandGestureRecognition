import math

import numpy as np


def compute_distance(x1: float, x2: float, y1: float, y2: float):
    return math.dist((x1, y1), (x2, y2))


def compute_normalized_angle(vec1, vec2):
    dot = np.dot(vec1, vec2)
    det = np.cross(vec1, vec2)

    angle = math.atan2(det, dot)

    # Normalize angle between 0 and 1
    return (angle + math.pi) / (2 * math.pi)


def compute_distances_angles_from_wrist(landmarks) -> list:
    distances = []
    angles = []

    MIDDLE_FINGER_MCP_INDEX = 8

    # copy landmarks to avoid modifying the original list
    landmarks = list(landmarks)

    wrist_landmarks = landmarks.pop(0)
    middle_finger_mcp_landmarks = landmarks[MIDDLE_FINGER_MCP_INDEX]

    reference_vector = (
        np.float32(wrist_landmarks.x - middle_finger_mcp_landmarks.x),
        np.float32(wrist_landmarks.y - middle_finger_mcp_landmarks.y),
    )

    for i, landmark in enumerate(landmarks):
        distances.append(
            np.float32(
                compute_distance(
                    landmark.x, wrist_landmarks.x, landmark.y, wrist_landmarks.y
                )
            )
        )

        if i == MIDDLE_FINGER_MCP_INDEX:
            angles.append(0.0)  # Skip middle finger MCP
            continue

        vector = (
            np.float32(wrist_landmarks.x - landmark.x),
            np.float32(wrist_landmarks.y - landmark.y),
        )

        angles.append(np.float32(compute_normalized_angle(vector, reference_vector)))

    return distances + angles
