from statistics import mode
import cv2
from keras.models import load_model
import numpy as np
import winsound

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

# parameters for loading data and images
detection_model_path = 'haarcascade_frontalface_default.xml'
emotion_model_path = 'fer2013_mini_XCEPTION.33-0.65.hdf5'
emotion_labels = {0:'ANGRY',1:'DISGUSTED',2:'SCARED',3:'HAPPY',
                4:'SAD',5:'SURPRISED',6:'NATURAL'}

face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
emotion_target_size = emotion_classifier.input_shape[1:3]
emotion_window = []
frame_window = 10

# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)
while True:
    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = face_detection.detectMultiScale(gray_image, 1.3, 5)
    for face_coordinates in faces:
        x1,y1,width,height = face_coordinates
        x1,y1,x2,y2 = x1,y1,x1+width,y1+height
        #x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        #emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)
        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_text = mode(emotion_window)
        except:
            continue
        if emotion_text == 'SCARED' or emotion_text == 'ANGRY':
            winsound.Beep(600, 1000)
        color = (0,0,255)
        cv2.rectangle(rgb_image,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.putText(rgb_image,emotion_text,(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2,cv2.LINE_AA)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    tr_image = cv2.resize(bgr_image, (1024,800), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('window_frame', tr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break