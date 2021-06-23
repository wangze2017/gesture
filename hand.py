import cv2
from mediapipe.python import solutions


class HandDetector:
    def __init__(self):
        self.static_image_mode = True
        self.max_num_hands = 1
        self.min_detection_confidence = 0.5
        self.min_tracking_confidence = 0.5
        self.hands = solutions.hands.Hands(self.static_image_mode,
                                           self.max_num_hands,
                                           self.min_detection_confidence,
                                           self.min_tracking_confidence, )
        self.draw = solutions.drawing_utils
        self.solutions = solutions

    def handMarks(self, image):
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image=imgRGB)
        hand_landmarks = results.multi_hand_landmarks
        if hand_landmarks:
            for lms in hand_landmarks:
                self.draw.draw_landmarks(image, lms, self.solutions.hands.HAND_CONNECTIONS)
        return image

    def rectangle(self, image, image_size=256, hand_size=0.5, area=None):
        if area is None:
            area = [(0, 0), (1, 1)]
        hand = image
        rectangle = image
        success = False
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image=imgRGB)
        hand_landmarks = results.multi_hand_landmarks
        h, w, c = image.shape
        if hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lms in hand_landmarks:
                for lm in lms.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                x_mean = (x_min + x_max) / 2
                y_mean = (y_min + y_max) / 2
                width = (x_max - x_min + y_max - y_min) / 4
                x_min = int(x_mean - width / hand_size)
                y_min = int(y_mean - width / hand_size)
                x_max = int(x_mean + width / hand_size)
                y_max = int(y_mean + width / hand_size)
            if x_min > area[0][0] * w and y_min > area[0][1] * h and x_max < area[1][0] * w and y_max < area[1][1] * h:
                success = True
                hand = image[y_min:y_max, x_min:x_max]
                hand = cv2.resize(hand, (image_size, image_size))
                rectangle = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        return hand, rectangle, success
