import cvzone
import cv2
from cvzone.ClassificationModule import Classifier

cap = cv2.VideoCapture(0)
myClassifier = Classifier('MyModel/keras_model.h5','MyModel/labels.txt')
fpsReader = cvzone.FPS()

while True:
    _, img = cap.read()
    predictions, index = myClassifier.getPrediction(img,scale=1.5)
    #print(predictions, index)
    fps, img = fpsReader.update(img,pos=(400,50))
    print(fps)

    cv2.imshow("Image",img)
    cv2.waitKey(1)