import cv2
import os
import imutils

person = 'Juanpa'
dataPath= "Recognition\data"
trainingVideosPath = "Recognition\VIDEOS4Training"
outPutVideoPath = dataPath + "\\" + person

if not os.path.exists(outPutVideoPath):
    print('Creando carpeta: ', outPutVideoPath)
    os.makedirs(outPutVideoPath)
    print('Carpeta creada.')

inputVideo = trainingVideosPath + "\\" + person + '.mp4'

if not os.path.exists(inputVideo):
    print('El video selecionado no existe.')
    exit(1)

cap = cv2.VideoCapture(inputVideo)
FACE_LIB = cv2.CascadeClassifier("lib\haarcascade_frontalface_default.xml")
MAX_FACES_NUMBER = 300
FACES_COUNT = 0

while True:
	ret,frame = cap.read()
    
	if ret == False:
		break
	
	frame = imutils.resize(frame, width=640)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	auxFrame = frame.copy()
    
	faces = FACE_LIB.detectMultiScale(gray, 1.1, 5)
    
	for (x,y,w,h) in faces:
		cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
		face = auxFrame[y:y+h, x:x+w]
		face = cv2.resize(face, (150,150), interpolation=cv2.INTER_CUBIC)
		outPutVideo = outPutVideoPath + "\\face_{}.jpg".format(FACES_COUNT)
		cv2.imwrite(outPutVideo, face)
		FACES_COUNT = FACES_COUNT + 1
	cv2.imshow('frame', frame)

	k = cv2.waitKey(1)
	if k == 27 or FACES_COUNT >= MAX_FACES_NUMBER:
		break
    
cap.release()
cv2.destroyAllWindows()