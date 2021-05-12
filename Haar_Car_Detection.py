import cv2
import argparse
import utils


def main():
    parser = argparse.ArgumentParser()
    # file path argument
    parser.add_argument("--path", type=str, default='../../data/traffic/Traffic_Road.mp4',
                        help="file path to file")
    parser.add_argument("--detect", type=str, default='car',
                        help="object to detect ooption [face, car]")
    args = parser.parse_args()

    directory, file, ext = utils.get_head_tail_ext(args.path)

    if 'mp4' in ext or 'avi' in ext:
        utils.detect_in_video(args.path, args.detect)
    elif 'jpg' in ext or 'png' in ext:
        utils.detect_in_image(args.path, args.detect)
    else:
        message = 'Extention not valid'
        utils.sys_exit(message)


if __name__ == '__main__':
    main()
