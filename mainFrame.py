import cv2
import os
import numpy as np
import tensorflow as tf

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

model_for_v = tf.keras.models.load_model('model.h5')
model_for_me = tf.keras.models.load_model('model_me.h5')

pic_taken = False

text2 = ""

def detectSign(roi_img):
	roi_img = np.expand_dims(roi_img, axis=0)
	prediction = model_for_v.predict(roi_img)
	prediction = np.argmax(prediction)
	return prediction
		
cap = cv2.VideoCapture(0)

box_size = 234
width = int(cap.get(3))

while True:
	ret, frame = cap.read()

	frame = cv2.flip(frame, 1)

	cv2.rectangle(frame, (width - box_size, 0), (width, box_size), (0, 250, 0), 2)

	roi = frame[5: box_size-5 , width-box_size + 5: width -5]
	roi_img = np.array(roi)

	prediction = detectSign(roi_img)

	faces = face_cascade.detectMultiScale(frame, 1.3 , 5)

	for (x,y,w,h) in faces:
		roi_face = frame[x:x+w , y:y+h]
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 250, 0), 2)
		cv2.imshow("face", roi_face)

	if prediction == 0:
		text1 = "V sign noticed"
		if not pic_taken:
			cv2.imwrite("full_frame.png", frame)
			cv2.imwrite("face_on_frame.png", roi_face)
			pic_taken = True
	else:
		text1 = "Show V to click a pic"

	h, w = 224, 224

	if pic_taken:
		frameImage = cv2.imread("face_on_frame.png")
		frameImage = cv2.cvtColor(frameImage, cv2.COLOR_BGR2RGB)
		frameImage = cv2.resize(frameImage, (h, w))
		frameImage = np.array(frameImage) / 255.0
		frameImage = np.expand_dims(frameImage, axis=0)

		predicted_frame = model_for_me.predict(frameImage)
		predicted_frame = np.argmax(predicted_frame)
		
		text2 = "Rohan" if predicted_frame == 0 else "Not Rohan"

	cv2.putText(frame, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 1, cv2.LINE_AA)
	cv2.putText(frame, text2, (3, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 1, cv2.LINE_AA)

	k = cv2.waitKey(1)

	if k == ord('q'):
		break

	cv2.imshow("Gesture", frame)


cap.release()
cv2.destroyAllWindows()