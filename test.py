from keras.preprocessing import image
from matplotlib import pyplot as plt
import numpy as np
import joblib as jb
import cv2

objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# Initialize video capture and load models
cap = cv2.VideoCapture("video.mp4")
face_detector = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
model = jb.load("model")
a = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (48, 48), (104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), thickness=1)

            # if a % 2 == 0:
            roi_gray = cv2.cvtColor(frame[y:y2, x:x2], cv2.COLOR_BGR2GRAY)
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255

            custom = model.predict(img_pixels)[0]
            print(custom)
            if(max(custom) > 0.6 or a == 0):
              emotion_label = objects[custom.argmax(axis=0)]
            cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    a += 1
    cv2.imshow('Video', frame)
    if cv2.waitKey(25) == 13:
        break
cap.release()
cv2.destroyAllWindows()
