# Import Modules
import cv2
import numpy as np
import face_recognition
import os
import pandas as pd

path = 'images'
images = []
classNames = []
#TOLERANCE = 0.9
myList = os.listdir(path)
print(myList)
for x in myList:
    subject_dir_path = path + "/" + x
    subject_images_names = os.listdir(subject_dir_path)
    for image_name in subject_images_names:
        image_path = subject_dir_path + "/" + image_name

        curImg = cv2.imread(f'{subject_dir_path}/{image_name}')
        images.append(curImg)
        classNames.append(image_name)

print(classNames)

'''
for x in range(len(images)):
    print(images[x])
'''
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

df=pd.read_csv("players.csv")
sorted_csv = df.sort_values(by=['player'])
df=sorted_csv
img_names = df.image.tolist()
img_players = df.player.tolist()

dictionary = dict(zip(img_names, img_players))

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex]
            for key in dictionary:
                if key==name:
                    print_name=dictionary[key]
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, print_name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)  #delay it for 1 millisecond