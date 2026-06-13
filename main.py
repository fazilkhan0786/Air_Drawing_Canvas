"""
AirDrawing main module.
Author attribution: Fazilkhan Malek.
This note is for documentation/backend purposes only and does not affect the algorithm or runtime behavior.
"""

import cv2
import numpy as np
import math
from hand_tracker import HandTracker
from gesture_controller import GestureController

xp, yp = 0, 0

canvas = np.zeros((720, 1280, 3), np.uint8)

draw_color = (255, 0, 255)

brush_thickness = 10
eraser_thickness = 50


def draw_toolbar(frame):

    colors = [
        ((255, 0, 255), "Purple"),
        ((255, 0, 0), "Blue"),
        ((0, 255, 0), "Green"),
        ((0, 0, 255), "Red"),
        ((0, 255, 255), "Yellow"),
        ((0, 165, 255), "Orange"),
        ((255, 255, 0), "Cyan"),
        ((255, 255, 255), "White"),
        ((0, 0, 0), "Black")
    ]

    box_width = 1280 // len(colors)

    for i, (color, name) in enumerate(colors):

        x1 = i * box_width
        x2 = (i + 1) * box_width

        cv2.rectangle(frame,
                      (x1, 0),
                      (x2, 80),
                      color,
                      -1)

        text_color = (0, 0, 0)

        if color == (0, 0, 0):
            text_color = (255, 255, 255)

        cv2.putText(frame,
                    name,
                    (x1 + 10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    text_color,
                    2)

    return frame


def main():

    global xp, yp
    global canvas
    global brush_thickness
    global draw_color

    cap = cv2.VideoCapture(0)

    cap.set(3, 1280)
    cap.set(4, 720)

    detector = HandTracker()

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        frame = draw_toolbar(frame)

        frame = detector.find_hands(frame)

        landmarks = detector.find_position(frame)

        if landmarks:

            x1, y1 = landmarks[8][1], landmarks[8][2]

            fingers = GestureController.fingers_up(landmarks)

            mode = GestureController.detect_mode(fingers)

            # COLOR SELECTION
            if mode == "SELECT" and y1 < 80:

                colors = [
                    (255, 0, 255),   # Purple
                    (255, 0, 0),     # Blue
                    (0, 255, 0),     # Green
                    (0, 0, 255),     # Red
                    (0, 255, 255),   # Yellow
                    (0, 165, 255),   # Orange
                    (255, 255, 0),   # Cyan
                    (255, 255, 255), # White
                    (0, 0, 0)        # Black
                ]

                box_width = 1280 // len(colors)

                selected = x1 // box_width

                if selected < len(colors):
                    draw_color = colors[selected]

                xp, yp = 0, 0

            cv2.putText(frame,
                        f"Mode: {mode}",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)

            cv2.putText(frame,
                        f"Thickness: {brush_thickness}",
                        (20, 170),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        2)

            # DRAW
            if mode == "DRAW":

                cv2.circle(frame,
                           (x1, y1),
                           15,
                           draw_color,
                           cv2.FILLED)

                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                cv2.line(canvas,
                         (xp, yp),
                         (x1, y1),
                         draw_color,
                         brush_thickness)

                xp, yp = x1, y1

            # THICKNESS CONTROL
            elif mode == "THICKNESS":

                tx, ty = landmarks[4][1], landmarks[4][2]

                cv2.circle(frame,
                           (tx, ty),
                           10,
                           (0, 255, 255),
                           cv2.FILLED)

                cv2.line(frame,
                         (tx, ty),
                         (x1, y1),
                         (0, 255, 255),
                         3)

                distance = math.hypot(
                    x1 - tx,
                    y1 - ty
                )

                new_thickness = int(
                    np.interp(
                        distance,
                        [40, 180],
                        [5, 40]
                    )
                )

                brush_thickness = int(
                    0.8 * brush_thickness +
                    0.2 * new_thickness
                )

                xp, yp = 0, 0

            # ERASER
            elif mode == "ERASE":

                cv2.circle(frame,
                           (x1, y1),
                           20,
                           (0, 0, 255),
                           cv2.FILLED)

                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                cv2.line(canvas,
                         (xp, yp),
                         (x1, y1),
                         (0, 0, 0),
                         eraser_thickness)

                xp, yp = x1, y1

            # CLEAR CANVAS
            elif mode == "CLEAR":

                canvas = np.zeros((720, 1280, 3), np.uint8)

                xp, yp = 0, 0

            else:
                xp, yp = 0, 0

        else:
            xp, yp = 0, 0

        gray = cv2.cvtColor(canvas,
                            cv2.COLOR_BGR2GRAY)

        _, inv = cv2.threshold(gray,
                               50,
                               255,
                               cv2.THRESH_BINARY_INV)

        inv = cv2.cvtColor(inv,
                           cv2.COLOR_GRAY2BGR)

        frame = cv2.bitwise_and(frame,
                                inv)

        frame = cv2.bitwise_or(frame,
                               canvas)

        cv2.imshow("Advanced Air Whiteboard",
                   frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            canvas = np.zeros((720, 1280, 3), np.uint8)

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()