import cv2
import argparse
import utils


def main():
    parser = argparse.ArgumentParser()
    # file path argument
    parser.add_argument("--path", type=str, default='../data/trafic/Traffic_Road.mp4',
                        help="file path to file")
    parser.add_argument("--detect", type=str, default='face',
                        help="object to detect ooption [face, car]")
    args = parser.parse_args()

    img_name = 'Window'
    cv2.namedWindow(img_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(img_name, 40, 30)
    cv2.resizeWindow(img_name, 400, 400)

    directory, file, ext = utils.get_head_tail_ext(args.path)

    if 'mp4' in ext or 'avi' in ext:
        capt = cv2.VideoCapture(args.path)
    elif 'jpg' in ext or 'png' in ext:
        img = cv2.imread(args.path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if args.detect == 'car':
        cars = utils.car_cascade(gray)
        for coor in cars:
            img = utils.draw_rect(img, coor)
    elif args.detect == 'face':
        faces, eyes = utils.face_cascade(gray)
        for coor in faces:
            img = utils.draw_rect(img, coor)
        for coor in eyes:
            img = utils.draw_rect(img, coor)

    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
