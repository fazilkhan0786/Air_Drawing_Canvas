class GestureController:

    @staticmethod
    def fingers_up(landmarks):

        fingers = []

        # Thumb
        if landmarks[4][1] < landmarks[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Index, Middle, Ring, Pinky
        tip_ids = [8, 12, 16, 20]

        for tip in tip_ids:
            if landmarks[tip][2] < landmarks[tip - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    @staticmethod
    def detect_mode(fingers):

        # DRAW: Only Index Finger
        if fingers == [0, 1, 0, 0, 0]:
            return "DRAW"

        # SELECT: Index + Middle
        elif fingers == [0, 1, 1, 0, 0]:
            return "SELECT"

        # THICKNESS: Thumb + Index
        elif fingers == [1, 1, 0, 0, 0]:
            return "THICKNESS"

        # ERASER: Fist
        elif fingers == [0, 0, 0, 0, 0]:
            return "ERASE"

        # CLEAR: All fingers except thumb
        elif fingers[1:] == [1, 1, 1, 1]:
            return "CLEAR"

        return "NONE"