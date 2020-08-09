import cv2
import os
import numpy as np
import tensorflow as tf

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

model_for_me = tf.keras.models.load_model('model_me.h5')

def detectPerson(roi_face):
	h, w = 224, 224
	roi_face = cv2.resize(roi_face, (h, w))
	roi_face = np.expand_dims(roi_face, axis=0)
	prediction = model_for_me.predict(roi_face)
	prediction = np.argmax(prediction)
	return prediction

cap = cv2.VideoCapture(0)

while True:
	ret, frame = cap.read()

	frame = cv2.flip(frame, 1)

	faces = face_cascade.detectMultiScale(frame, 1.3 , 5)

	text = ""

	for (x,y,w,h) in faces:
		roi_face = frame[x:x+w , y:y+h]

		prediction = detectPerson(roi_face / 255.0)

		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 250, 0), 2)
		cv2.imshow("face", roi_face)

		text = "Rohan" if prediction == 0 else "Not Rohan!"
		
		print(prediction)

	cv2.putText(frame, str(text), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 1, cv2.LINE_AA)

	k = cv2.waitKey(1)

	if k == ord('q'):
		break

	cv2.imshow("frame", frame)
	

cap.release()
cv2.destroyAllWindows()