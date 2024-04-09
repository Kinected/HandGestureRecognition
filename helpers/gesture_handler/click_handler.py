import time
from enum import Enum

MAX_HOLDING_SINCE = 1.5


class States(Enum):
    PENDING = 1
    FIRST_OPEN = 2
    FIRST_CLOSED = 3
    SECOND_OPEN = 4
    SECOND_CLOSED = 5
    CLICK = 6


class ClickHandler:
    current_state = States.PENDING
    previous_state = None
    state_since = None

    current_gesture = None
    previous_gesture = None

    def handle_step(self, current_gesture):
        """
        Function that handles the current step of a click
        :param current_gesture: The current gesture
        :return:
        """
        if current_gesture == "palm" or current_gesture == "closed":
            self.previous_gesture = self.current_gesture
            self.current_gesture = current_gesture

        current_time = time.time()
        holding_since = current_time - self.state_since if self.state_since else None

        if holding_since and holding_since > MAX_HOLDING_SINCE:
            self.state_since = current_time
            self.current_state = States.PENDING

        transitions = {
            States.PENDING: {
                "condition": current_gesture == "palm",
                "next_state": States.FIRST_OPEN
            },
            States.FIRST_OPEN: {
                "condition": self.previous_gesture == "palm" and current_gesture == "closed" and holding_since <= MAX_HOLDING_SINCE,
                "next_state": States.FIRST_CLOSED
            },
            States.FIRST_CLOSED: {
                "condition": self.previous_gesture == "closed" and current_gesture == "palm" and holding_since <= MAX_HOLDING_SINCE,
                "next_state": States.SECOND_OPEN
            },
            States.SECOND_OPEN: {
                "condition": self.previous_gesture == "palm" and current_gesture == "closed" and holding_since <= MAX_HOLDING_SINCE,
                "next_state": States.SECOND_CLOSED
            },
            States.SECOND_CLOSED: {
                "condition": self.previous_gesture == "closed" and current_gesture == "palm" and holding_since <= MAX_HOLDING_SINCE,
                "next_state": States.CLICK
            },
            States.CLICK: {
                "condition": True,
                "next_state": States.PENDING
            }
        }

        if self.current_state not in transitions:
            self.state_since = current_time
            self.current_state = States.PENDING

        transition = transitions[self.current_state]

        if transition["condition"]:
            self.state_since = current_time
            self.current_state = transition["next_state"]

    def is_clicking(self):
        return self.current_state == States.CLICK
