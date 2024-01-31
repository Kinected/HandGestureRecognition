import numpy as np

from helpers.computations import compute_distances_angles_from_wrist


def get_label(labels, predictions: list[float]):
    if np.all(predictions == predictions[0]):
        return "none"

    index = np.argmax(predictions)

    return labels[index]


def get_gesture(model, labels: list[str], min_confidence: float, hand, landmarks):
    landmarks_distances_and_angles = compute_distances_angles_from_wrist(
        landmarks.landmark
    )

    input_data = np.concatenate((hand, landmarks_distances_and_angles))
    input_data = input_data.reshape(1, -1)

    predictions = model.predict(input_data, verbose=0)

    accuracy = np.max(predictions)

    if accuracy < min_confidence:
        return "no_gesture", 0

    return (get_label(labels, predictions[0]), accuracy)
