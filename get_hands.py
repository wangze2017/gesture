"""
get hand pictures from videos
"""
import os
import cv2
from hand import HandDetector


def get_hand(videodirname, imagedirname, imagename, area=None):
    videos = []
    if not os.path.exists(imagedirname):
        os.makedirs(imagedirname)
    if area is None:
        area = [(0, 0), (1, 0.75)]
    detector = HandDetector()
    for i in os.listdir(videodirname):
        videos.append(os.path.join(videodirname, i))
    count = 0
    for video in videos:
        cap = cv2.VideoCapture(video)
        while True:
            ret, cv2_image = cap.read()
            if not ret:
                break
            hand, rectangle, success = detector.rectangle(cv2_image, area=area)
            if success:
                hand = cv2.resize(hand, (256, 256))
                count = count + 1
                print(f'{imagedirname}/{imagename}_{count}.png')
                cv2.imwrite(f'{imagedirname}/{imagename}_{count}.png', hand)
    return


if __name__ == '__main__':
    for i in os.listdir('videos'):
        if i != '.DS_Store':
            print(i)
            get_hand(videodirname=f'videos/{i}', imagedirname=f'hands/{i}', imagename=i)
