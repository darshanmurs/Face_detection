# Importing Required Libraries

import cv2
import numpy as np
import face_recognition as face_rec


# Image Resizing Function

def resize(img, size):
    img_width = int(img.shape[1]* size)
    img_height = int(img.shape[0]*size)
    dimension = (img_width, img_height)
    return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)

# Image Encoding

img1 = face_rec.load_image_file(r'Images\Darshan.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img1 = resize(img1, 0.7)
img2 = face_rec.load_image_file(r'Images\Rohit Sharma.jpg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img2 = resize(img2, 0.7)

# finding face location

faceLoc= face_rec.face_locations(img1)[0]
img1_encode = face_rec.face_encodings(img1)[0]
cv2.rectangle(img1, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 3)

faceLoc2= face_rec.face_locations(img2)[0]
img2_encode = face_rec.face_encodings(img2)[0]
cv2.rectangle(img2, (faceLoc2[3], faceLoc2[0]), (faceLoc2[1], faceLoc2[2]), (255, 0, 255), 3)

# Comparing Images
img_compare = face_rec.compare_faces([img1_encode],img2_encode)
print(img_compare)
cv2.putText(img2, f'{img_compare}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255), 2 )

cv2.imshow('main_img', img1)
cv2.imshow('test_img', img2)
cv2.waitKey(0)
cv2.destroyWindow()
