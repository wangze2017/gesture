import cv2
import os
import paddlex.deploy


class predict:
    def __init__(self, model):
        self.model = paddlex.deploy.Predictor(model, use_gpu=True)
        self.model_type = self.model.model_type

    def result(self, image):
        image = image.astype('float32')
        image = cv2.resize(image, (288, 288))
        result = self.model.predict(image)
        return result[0]

if __name__ == "__main__":
    model = predict(model='D:\Codes\Python\gesture\models\inference_model1')
    image = cv2.imread('hands/bu/bu61.png')
    print(model.result(image))








