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

    wrist_landmarks = landmarks.pop(0)  # Get and remove wrist landmark
    middle_finger_mcp_landmarks = landmarks[MIDDLE_FINGER_MCP_INDEX]

    reference_vector = (
        wrist_landmarks.x - middle_finger_mcp_landmarks.x,
        wrist_landmarks.y - middle_finger_mcp_landmarks.y,
    )  # Vector from wrist to middle finger MCP

    for i, landmark in enumerate(landmarks):
        distances.append(
            compute_distance(
                landmark.x, wrist_landmarks.x, landmark.y, wrist_landmarks.y
            )
        )  # Distance from wrist to landmark

        if i == MIDDLE_FINGER_MCP_INDEX:
            angles.append(0.0)  # Skip middle finger MCP
            continue

        # compute angle between wrist-middle finger MCP vector and wrist-landmark vector
        vector = (
            wrist_landmarks.x - landmark.x,
            wrist_landmarks.y - landmark.y,
        )

        angles.append(compute_normalized_angle(vector, reference_vector))

    return distances + angles
