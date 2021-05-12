import os
import cv2


def get_head_tail_ext(path):
    ext = None
    dir_path, file_name = os.path.split(path)
    if file_name:
        file_name, ext = os.path.splitext(file_name)
    return dir_path, file_name, ext


def face_cascade(img):
    face_file = 'haarcascade_frontalface_default.xml'
    eye_file = 'haarcascade_eye.xml'
    face_cascade = cv2.CascadeClassifier(face_file)
    eye_cascade = cv2.CascadeClassifier(eye_file)
    faces = face_cascade.detectMultiScale(img)
    face_coor = list()
    eye_coor = list()
    for (x, y, w, h) in faces:
        face_coor.append([x, y, w, h])
        roi = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi)
        for (ex, ey, ew, eh) in eyes:
            eye_coor.append([ex, ey, ew, eh])
    return face_coor, eye_coor


def car_cascade(img):
    car_file = 'car.xml'
    eye_file = 'haarcascade_eye.xml'
    car_cascade = cv2.CascadeClassifier(car_file)
    cars = car_cascade.detectMultiScale(img)
    cars_coor = list()
    for (x, y, w, h) in cars:
        cars_coor.append([x, y, w, h])
    return cars_coor


def resize(img, factor=1):
    height, width = img.shape[:2]
    if factor == 1 and height > 400:
        factor = 400 / height
    w = int(width * factor)
    h = int(height * factor)
    return cv2.resize(img, (w, h))


def draw_rect(img, coor, color=(255, 0, 0), thicknes=2):
    x, y, w, h = coor
    return cv2.rectangle(img, (x, y), (x + w, y + h), color, thicknes)
