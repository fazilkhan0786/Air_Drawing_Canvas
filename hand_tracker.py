import os
import cv2
import mediapipe as mp


class HandTracker:
    def __init__(self):
        # Create HandLandmarker using the new API
        model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        self.hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.results = None

    def find_hands(self, img, draw=True):
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Detect hand landmarks
        self.results = self.hand_landmarker.detect(mp_image)
        
        # Draw landmarks if requested
        if draw and self.results.hand_landmarks:
            for hand_landmarks in self.results.hand_landmarks:
                # Draw on image
                h, w, _ = img.shape
                for landmark in hand_landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

        return img

    def find_position(self, img):
        landmarks = []

        if self.results and self.results.hand_landmarks:
            hand = self.results.hand_landmarks[0]
            h, w, _ = img.shape

            for idx, lm in enumerate(hand):
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                landmarks.append([idx, cx, cy])

        return landmarks