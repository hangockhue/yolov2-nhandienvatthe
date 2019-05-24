import time
import dlib
import cv2

#

image = cv2.imread("./blackpink.jpg")
cv2.imshow("image", image)
cv2.waitKey(0)
hog_face_detector = dlib.get_frontal_face_detector()

#cnn_face_detector = dlib.cnn_face_detection_model_v1("./mmod_human_face_detector.dat")

start = time.time()
face_hog = hog_face_detector(image, 1)
end = time.time()
print("Quá trình thực hiện Hog + SVM là " + str(end-start))

for face in face_hog:
    x = face.left()
    y = face.top()
    w = face.right() - x
    h = face.bottom() - y

    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
'''
start = time.time()
face_cnn = cnn_face_detector(image, 1)
end = time.time()
print("Quá trình thực hiện CNN" + str(end-start))

for face in face_hog:
    x = face.left()
    y = face.top()
    w = face.right() - x
    h = face.bottom() - y

    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
'''
cv2.imshow("image", image)
cv2.waitKey(0)
