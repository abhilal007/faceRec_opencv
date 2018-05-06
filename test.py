import cv2
import matplotlib.pyplot as plt
import time


def convertTORGB(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
test = cv2.imread('img/test.jpg')
gray_img = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_img, cmap='gray')
haar_face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5);  
 
#print the number of faces found 
print('Faces found: ', len(faces))

for (x, y, w, h) in faces:     
         cv2.rectangle(test, (x, y), (x+w, y+h), (0, 255, 0), 2)
         
cv2.imshow('frame',test)
cv2.waitKey(6000)

