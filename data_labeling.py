import argparse
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from shutil import copy2

# allowed inputs
VALID_INPUTS = ['b', 'g', 'e', 'u', 'd']
# directories where files will be saved according to selection
GOOD_DIR = '1_'
BAD_DIR = '0_'
UNSURE_DIR = '2_'


def main(args):
    # init
    print('Image labeling')
    print('Init...')
    print('Creating folder structure...')
    create_struct(args.t, args.p)
    print('Done.')
    img_files = glob.glob(os.path.join(args.s, "*.jp*"))
    print(f'# images to label: {len(img_files)}')
    print('Instructions:')
    print('g - image is a good image')
    print('b - image is a bad image')
    print('u - unsure about image')
    print('d - do not copy image')
    print('e - exit this script')
    for idx, img_file in enumerate(img_files):
        print(f"{idx + 1}/{len(img_files)}")
        show_image(img_file)
        sel = selection()
        if sel == 'e':
            exit(0)
            break
        elif sel == 'd':
            os.remove(img_file)
            continue
        folder_decision = folder_selection(sel)
        copy2(img_file, os.path.join(args.t, folder_decision+args.p))
        os.remove(img_file)


def create_struct(trg_dir, pose):
    """
    Creates directory structure where labeled images will be saved.
    :param trg_dir: directory under which structure will be created
    """
    # try:
    #     os.mkdir(trg_dir + pose)
    # except FileExistsError as e:
    #     pass
    try:
        os.mkdir(os.path.join(trg_dir, GOOD_DIR + pose))
    except FileExistsError as e:
        pass
    try:
        os.mkdir(os.path.join(trg_dir, BAD_DIR + pose))
    except FileExistsError as e:
        pass
    try:
        os.mkdir(os.path.join(trg_dir, UNSURE_DIR + pose))
    except FileExistsError as e:
        pass


def show_image(img_file):
    """
    Reads and show an image file.
    :param img_file: File to read and show
    """
    img = mpimg.imread(img_file)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.show(block=False)


def folder_selection(sel):
    """
    Selects the folder based on user input.
    :param sel: user input
    """
    if sel == 'b':
        return BAD_DIR
    elif sel == 'g':
        return GOOD_DIR
    elif sel == 'u':
        return UNSURE_DIR


def selection():
    """
    Gets user input.
    :return: user input
    """
    sel = None
    while sel not in VALID_INPUTS:
        sel = input("[g | b | u | d| e]: ")
        if sel not in VALID_INPUTS:
            print(f"Your selection {sel} is not in valid inputs {VALID_INPUTS}!")
    return sel


def parse_args():
    parser = argparse.ArgumentParser(description='Data labeling tool.')
    parser.add_argument("-s", type=str,
                        required=True, help="Directory that contains the images to label.")
    parser.add_argument("-t", type=str,
                        required=True, help="Directory that will contain the labeled images.")
    parser.add_argument("-p", type=str,
                        required=True, choices={"downwardDog", "warrior1", "warrior2"}, help="Name of Pose you are classfying")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # args = argparse.Namespace(s="Good Yoga/Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_", t="test")
    main(args)
