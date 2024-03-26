import cv2

BOX_MARGIN = 24


def draw_box(frame, gesture, accuracy, hand, landmarks, box_margin=BOX_MARGIN):
    x_min = 10000
    x_max = 0
    y_min = 10000
    y_max = 0

    for landmark in landmarks.landmark:
        x = landmark.x * frame.shape[1]
        y = landmark.y * frame.shape[0]

        if x < x_min:
            x_min = x - box_margin
        if x > x_max:
            x_max = x + box_margin
        if y < y_min:
            y_min = y - box_margin
        if y > y_max:
            y_max = y + box_margin

    color = (0, 255, 255)

    if gesture == "palm":
        color = (0, 255, 0)

    if gesture == "closed":
        color = (0, 0, 255)

    frame = cv2.rectangle(
        frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 3
    )
    frame = cv2.putText(
        frame,
        gesture + " " + str(accuracy),
        (int(x_min), int(y_max + box_margin)),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        color,
        2,
        cv2.LINE_AA,
    )

    frame = cv2.putText(
        frame,
        "left" if hand[0] == 1 else "right",
        (int(x_min), int(y_min - box_margin)),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        color,
        2,
        cv2.LINE_AA,
    )

    return frame


def get_landmarks(frame, holistics):
    frame.flags.writeable = False
    results = holistics.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame.flags.writeable = True

    return {
        "left": results.left_hand_landmarks or None,
        "right": results.right_hand_landmarks or None,
    }


def draw_landmarks(frame, landmarks, holistics, drawing):
    drawing.draw_landmarks(
        frame,
        landmarks,
        holistics.HAND_CONNECTIONS,
        drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
        drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=2),
    )

    return frame
