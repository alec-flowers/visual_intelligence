import argparse
import os
import glob
import cv2.cv2 as cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pynput import keyboard
from shutil import copy2

valid_inputs = ['b', 'g', 'e', 'u']
valid_inputs_noenter = [keyboard.KeyCode.from_char('b'), keyboard.KeyCode.from_char('g'),
                        keyboard.KeyCode.from_char('e'), keyboard.KeyCode.from_char('u')]

GOOD_DIR = 'good'
BAD_DIR = 'bad'
UNSURE_DIR = 'unsure'


def main(args):
    create_struct(args.t)
    img_files = glob.glob(os.path.join(args.s, "*.jpg"))
    for img_file in img_files:
        # img = cv2.imread(img_file)
        # img = resize_img(img)
        # # image = cv2.resize(image, (500, 500))
        # cv2.imshow('Mediapipe Feed', img)
        # cv2.waitKey()
        show_image(img_file)
        # sel = selection_noenter()
        # if sel == keyboard.KeyCode.from_char('e'):
        #     break
        # plt.close()
        sel = selection()
        if sel == 'e':
            exit(0)
        folder_decision = folder_selection(sel)
        copy2(img_file, os.path.join(args.t, folder_decision))


def create_struct(trg_dir):
    try:
        os.mkdir(trg_dir)
    except FileExistsError as e:
        pass
    try:
        os.mkdir(os.path.join(trg_dir, GOOD_DIR))
    except FileExistsError as e:
        pass
    try:
        os.mkdir(os.path.join(trg_dir, BAD_DIR))
    except FileExistsError as e:
        pass
    try:
        os.mkdir(os.path.join(trg_dir, UNSURE_DIR))
    except FileExistsError as e:
        pass


def show_image(img_file):
    img = mpimg.imread(img_file)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.show(block=False)


def resize_img(img):
    height, width = img.shape[:2]
    max_height = 400
    max_width = 400

    # only shrink if img is bigger than required
    if max_height < height or max_width < width:
        # get scaling factor
        scaling_factor = max_height / float(height)
        if max_width / float(width) < scaling_factor:
            scaling_factor = max_width / float(width)
        # resize image
        img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return img


def folder_selection(sel):
    if sel == 'b':
        return BAD_DIR
    elif sel == 'g':
        return GOOD_DIR
    elif sel == 'u':
        return UNSURE_DIR


def selection_noenter():
    with keyboard.Events() as events:
        while True:
            # Block for as much as possible
            event = events.get()
            if event.key not in valid_inputs_noenter:
                print(f"Your selection {event.key} is not in valid inputs {valid_inputs_noenter}")
            else:
                break
    return event.key


def selection():
    sel = None
    while sel not in valid_inputs:
        sel = input("You select: ")
        if sel not in valid_inputs:
            print(f"Your selection {sel} is not in valid inputs {valid_inputs}")
    return sel


def parse_args():
    parser = argparse.ArgumentParser(description='Data labeling tool.')
    parser.add_argument("-s", "--source_directory", type=str,
                        required=True, help="Directory that contains the images to label.")
    parser.add_argument("-t", "--target_directory", type=str,
                        required=True, help="Directory that will contain the labeled images.")
    return parser.parse_args()


if __name__ == '__main__':
    # args = parse_args()
    args = argparse.Namespace(s="Good Yoga/Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_", t="test")
    main(args)
