import cv2

face_cascade = cv2.CascadeClassifier("lib\haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while True:
	_, img = cap.read()
	if img is not None:
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	else:
		print("Error al cargar la imagen.")
		break

	faces = face_cascade.detectMultiScale(gray, 1.1, 4)
	for (x, y, w, h) in faces:
		cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
	cv2.imshow('img', img)
	key = cv2.waitKey(30)
	if key == 27:
		break
cap.release()