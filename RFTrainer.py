import cv2
import os
import numpy as np

dataPath = 'Recognition/data'
peopleList = os.listdir(dataPath)
print ('Lista de personas: ' , peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print('Leyendo Imagenes')

    for fileName in os.listdir(personPath):
        print('Rostros: ', nameDir + '/' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(personPath +'/'+fileName,0))
        image = cv2.imread(personPath +'/'+fileName,0)
        cv2.imshow('img', image)
        cv2.waitKey(10)
    label = label + 1

#Aquí verificamos que labels tenga 300 ceros
# uno por cada imagen de la carpeta Juanpa dentro de data
#entre mas videos de personas, se va sumando 1 a labels.
    
print('labels= ', labels)
print('Número de etiquetas 0:', np.count_nonzero(np.array(labels) ==0))
print(facesData)
cv2.destroyAllWindows

#Explicacion de algorithmos: https://upcommons.upc.edu/bitstream/handle/2117/331277/TFG_An%C3%A1lisis%20de%20un%20sistema%20de%20reconocimiento%20facial%20a%20partir%20de%20una%20base%20de%20datos%20realizado%20mediante%20Python.pdf

#NOTA! en caso de encontrar errores al ejecutar: cv2.face.EigenFaceRecognizer_create()
#Seguir estos pasos:

#uninstall opencv-python
#-pip uninstall opencv-python
#then install opencv-contrib-python
#-pip install opencv-contrib-python
#print(dir (cv2.face))

face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
#face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# Entrenando el reconocedor de rostros

print("Entrenando...")
face_recognizer.train(facesData, np.array(labels))

# Almacenando el modelo obtenido
face_recognizer.write('modeloEigenFace.xml')
#face_recognizer.write('modeloFisherFace.xml')
#face_recognizer.write('modeloLBPHFace.xml')
print("Modelo almacenado...")