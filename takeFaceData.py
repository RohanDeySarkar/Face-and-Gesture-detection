import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

counter = 100


while True:
	ret, frame = cap.read()

	frame = cv2.flip(frame, 1)

	faces = face_cascade.detectMultiScale(frame, 1.3 , 5)

	for (x,y,w,h) in faces:
		cv2.rectangle(frame, (x,y), (x+w , y+h),(0, 250, 0), 2)
		roi = frame[x:x+w , y:y+h]

	k = cv2.waitKey(1)

	if k == ord('v'):
		for i in range(counter):
			cv2.imwrite("data/detectPerson/rohan_data/" + str(i) + ".png", roi)
	if k == ord('b'):
		for i in range(counter):
			cv2.imwrite("data/detectPerson/intruder_data/" + str(i) + ".png", frame)
	if k == ord('q'):
		break

	cv2.imshow("Gesture", frame)



cap.release()
cv2.destroyAllWindows()
