import time
from enum import Enum



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

        current_time = time.time()
        holding_since = self.state_since - current_time if self.state_since else None

        transitions = {
            States.PENDING: {
                "condition": current_gesture == "palm",
                "next_state": States.FIRST_OPEN
            },
            States.FIRST_OPEN: {
                "condition": self.previous_gesture == "palm" and current_gesture == "closed" and holding_since <= 0.75,
                "next_state": States.FIRST_CLOSED
            },
            States.FIRST_CLOSED: {
                "condition": self.previous_gesture == "closed" and current_gesture == "palm" and holding_since <= 0.75,
                "next_state": States.SECOND_OPEN
            },
            States.SECOND_OPEN: {
                "condition": self.previous_gesture == "palm" and current_gesture == "closed" and holding_since <= 0.75,
                "next_state": States.SECOND_CLOSED
            },
            States.SECOND_CLOSED: {
                "condition": self.previous_gesture == "closed" and current_gesture == "palm" and holding_since <= 0.75,
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

        return self.current_state
