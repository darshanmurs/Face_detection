import cv2
import numpy as np
import face_recognition as face_rec
import os
from datetime import datetime
import pyttsx3

try:
    def resize(img, size):
        img_width = int(img.shape[1] * size)
        img_height = int(img.shape[0] * size)
        dimension = (img_width, img_height)
        return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)


    folderPath = 'Images'
    imgList = os.listdir(folderPath)
    img = []
    imgName = []

    for file in imgList:
        curimg = cv2.imread(os.path.join(folderPath, file))
        img.append(curimg)
        imgName.append(os.path.splitext(file)[0])


    def findEncoding(images):
        imgEncode = []
        for img in images:
            img = resize(img, 0.50)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodedImg = face_rec.face_encodings(img)[0]
            imgEncode.append(encodedImg)
        return imgEncode


    encode_list = findEncoding(img)

    # Initialize text-to-speech engine
    engine = pyttsx3.init()

    # Initialize date tracking variables for each person
    current_dates = {name: None for name in imgName}
    unknown_face_detected = False


    def MarkAttendance(name):
        global unknown_face_detected
        global current_dates

        now = datetime.now()
        dateStr = now.strftime('%d-%b-%Y')

        with open('Attendance.csv', 'a+') as f:
            dataList = f.readlines()
            nameList = [entry.split(',')[0] for entry in dataList]

            if name not in nameList:
                last_date = current_dates.get(name, None)
                if dateStr != last_date:
                    current_dates[name] = dateStr  # Update the current date for the person
                    timeStr = now.strftime('%I:%M %p')  # Format time in 12-hour with AM/PM
                    f.write(f'\n{name},{dateStr},{timeStr}\n')  # Add newline character
                    statement = f'Face Detection Successful!, Marking Attendance {name}'
                    engine.say(statement)
                    engine.runAndWait()
                    unknown_face_detected = False  # Reset the flag when attendance is marked

        if not unknown_face_detected:
            # If the face is in the list, and an unknown face has not been detected before
            statement = 'Unknown Face or Unregistered face detected'
            engine.say(statement)
            engine.runAndWait()
            unknown_face_detected = True


    print("Opening Web Cam...")
    webcam = cv2.VideoCapture(0)
    while True:
        success, frame = webcam.read()
        small_frames = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        facesInWebcam = face_rec.face_locations(small_frames)
        encodeFacesInWebcam = face_rec.face_encodings(small_frames, facesInWebcam)

        for face, loc in zip(encodeFacesInWebcam, facesInWebcam):
            face_match = face_rec.compare_faces(encode_list, face)
            face_dist = face_rec.face_distance(encode_list, face)
            print(face_dist)
            matchIndex = np.argmin(face_dist)

            if face_match[matchIndex]:
                name = imgName[matchIndex].upper()
                y1, x1, y2, x2 = loc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.rectangle(frame, (x1, y2 - 25), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (x1 - 150, y2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
                MarkAttendance(name)
            else:
                y1, x1, y2, x2 = loc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.rectangle(frame, (x1, y2 - 25), (x2, y2), (0, 0, 255), cv2.FILLED)  # Red rectangle for unknown face
                cv2.putText(frame, 'Unknown', (x1 - 150, y2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)

        cv2.imshow('WebCam', frame)
        cv2.waitKey(1)
    video_capture.release()


except KeyboardInterrupt:
    print("Web cam closed")