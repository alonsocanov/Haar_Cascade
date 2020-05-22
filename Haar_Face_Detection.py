import cv2
import os


def main(file_name):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    img = cv2.imread(file_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey),
                          (ex + ew, ey + eh), (0, 255, 0), 2)
    img_name = "Faces"
    cv2.namedWindow(img_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(img_name, 40, 30)
    cv2.resizeWindow(img_name, 400, 400)
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # path to images
    img_path = "../data/faces/"
    # obtain images names from data where name terminates with .jpg
    img_names = [img_path + x for x in os.listdir(img_path) if ".jpg" in x]
    main(img_names[0])
