import os
import cv2
import sys


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
    car_file = 'cars.xml'
    car_cascade = cv2.CascadeClassifier(car_file)
    cars = car_cascade.detectMultiScale(img, 1.1, 1)
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


def webcam_avalability(webcam: cv2.VideoCapture):
    if not webcam.isOpened():
        sys_exit("Error opening webcam")


def check(c='q') -> bool:
    if cv2.waitKey(1) & 0xFF == ord(c):
        return True
    return False


def sys_exit(message):
    print(message)
    sys.exit(1)


def detect_in_video(path, detect):
    img_name = 'Window'
    cv2.namedWindow(img_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(img_name, 40, 30)
    cv2.resizeWindow(img_name, 400, 400)

    video = cv2.VideoCapture(path)
    webcam_avalability(video)
    q = False
    while video.isOpened() and not q:
        ret, frame = video.read()
        frame = resize(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if detect == 'car':
            cars = car_cascade(gray)
            for coor in cars:
                frame = draw_rect(frame, coor)
        elif detect == 'face':
            faces, eyes = face_cascade(gray)
            for coor in faces:
                img = draw_rect(img, coor)
            for coor in eyes:
                img = draw_rect(img, coor)
        else:
            message = 'Could not find detector'
            sys_exit(message)

        cv2.imshow(img_name, frame)

        q = check('q')
    video.release()
    cv2.destroyAllWindows()


def detect_in_image(path, detect):
    img_name = 'Window'
    cv2.namedWindow(img_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(img_name, 40, 30)
    cv2.resizeWindow(img_name, 400, 400)

    img = cv2.imread(path)
    img = resize(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if detect == 'car':
        cars = car_cascade(gray)
        for coor in cars:
            img = draw_rect(img, coor)
    elif detect == 'face':
        faces, eyes = face_cascade(gray)
        for coor in faces:
            img = draw_rect(img, coor)
        for coor in eyes:
            img = draw_rect(img, coor)
    else:
        message = 'Could not find detector'
        sys_exit(message)

    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
