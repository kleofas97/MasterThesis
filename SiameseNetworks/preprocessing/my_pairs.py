import os
import numpy as np
import cv2
from typing import Tuple
import itertools
import time
import SiameseNetworks.preprocessing.pairs_testing as tests

MARGIN = 20
TIMEOUT = 1
S_THRESHOLD = 0.98
"""My pairs module is provided for preparing a dataset of patches for model to learn.
    It may prepare and save a dataset with function: prepare_dataset_on_disk;
    or load it into directly into the memory with function: prepare_dataset_in_memory. 

"""


def prepare_dataset_on_disk(dataset_path_train: str, dataset_path_val: str, path_to_output: str,
                            train_set_size: int, val_set_size: int, patch_size: int):
    """Function to prepare dataset for future generator use. Please use generator provided in model.py
    :argument
        dataset_path_train (str) - path to folder with images for train patches
        dataset_path_val (str) - path to folder wth images for val patches
        path_to_output (str) - path to folder where patches from train and val sets will be stored
        train_set_size (int) - number of pairs of patches to be generated from images in train_folder
        val_set_size (int) - number of pairs of patches to be generated from images in val_folder
        patch_size (int) - size of patch to be generated
        s (int) - similarity level
    :return None; saves patches in provided directories
        """
    # 1 preparing path for train and validation dataset
    train_path, val_path = make_dirs(path_to_output)
    # 2 preparing train_set
    percent = int((train_set_size / 100))
    for i in range(train_set_size):
        if i % percent == 0:
            print('Train Set finished in {}%'.format(100 * i / train_set_size))
        p1, p2, label = get_random_pair(dataset_path_train, patch_size)
        p1 = p1.reshape(p1.shape[0], p1.shape[1], 1)
        p2 = p2.reshape(p2.shape[0], p2.shape[1], 1)
        cv2.imwrite(
            os.path.join(os.path.join(train_path, "train_0"), "{}_{}.png".format(i, label)), p1)
        cv2.imwrite(
            os.path.join(os.path.join(train_path, "train_1"), "{}_{}.png".format(i, label)), p2)
    # 3 preparing val_set
    percent = int(val_set_size / 100)
    for i in range(val_set_size):
        if i % percent == 0:
            print('Val Set finished in {}%'.format(100 * i / val_set_size))
        p1, p2, label = get_random_pair(dataset_path_val, patch_size)
        p1 = p1.reshape(p1.shape[0], p1.shape[1], 1)
        p2 = p2.reshape(p2.shape[0], p2.shape[1], 1)
        cv2.imwrite(os.path.join(os.path.join(val_path, "val_0"), "{}_{}.png".format(i, label)),
                    p1)
        cv2.imwrite(os.path.join(os.path.join(val_path, "val_1"), "{}_{}.png".format(i, label)),
                    p2)


def prepare_dataset_in_memory(folderName: str, set_size: int, patch_size: int) -> Tuple:
    """
    Loading the data to memory - only used when not using generator and previously prepared dataset
    :param folderName: foldername which contains whole pages
    :param set_size: set size
    :param patch_size: size of patches
    :return: Tuple of patches with labels; size depends on set_size
    """
    pairs = []
    labels = []
    percent = set_size / 100
    for i in range(set_size):
        # if i % percent  == 0:
        #     print('Set finished in {}%'.format(100 * i / set_size))
        p1, p2, label = get_random_pair(folderName, patch_size)
        p1 = p1.reshape(p1.shape[0], p1.shape[1], 1)
        p2 = p2.reshape(p2.shape[0], p2.shape[1], 1)
        pairs += [[p1, p2]]
        labels += [label]
    apairs = np.array(pairs, dtype=object)
    print(apairs.shape)
    alabels = np.array(labels)
    return apairs, alabels


def get_position(img: np.ndarray, patch_size: int) -> Tuple:
    """
    Find random position for patches within acceptable location
    :param img: image from which location of patches will be taken
    :param patch_size: size of desired patch
    :return: top-left point of patch p1 and patch p2
    """
    assert patch_size * 2 < img.shape[
        0], "Patch size to big, img vertical size is {}, while proposed patch {}. Reduce patch size".format(
        img.shape[0], patch_size)
    assert patch_size < img.shape[1], "Width of patch is to big"
    pos = [np.random.randint(low=0 + MARGIN, high=img.shape[0] - 2 * patch_size - MARGIN),
           np.random.randint(low=0 + MARGIN, high=img.shape[1] - patch_size - MARGIN)
           ]
    p1_pos = pos
    p2_pos = [pos[0] + patch_size, pos[1]]
    return p1_pos, p2_pos


def evaluate_s(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Based on two patches evaluate s value
    :param p1: patch nb 1
    :param p2: patch nb 2
    :return: s value (similarity)
    """
    _, p_th1 = cv2.threshold(p1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, p_th2 = cv2.threshold(p2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pixels1 = cv2.countNonZero(p_th1)
    pixels2 = cv2.countNonZero(p_th2)
    if pixels1 != 0 or pixels2 != 0:
        s = min(pixels1, pixels2) / max(pixels1, pixels2)
        return s
    else:
        return 1


def get_patches(img: np.ndarray, patch_size: int) -> Tuple:
    # """"""
    """
    Generate patches from image
    :param img: image to take patches from
    :param patch_size: size of patch
    :return: two patches; p1 is the above one and p2 is the below one
    """
    p1_pos, p2_pos = get_position(img, patch_size)
    p1 = img[p1_pos[0]:p1_pos[0] + patch_size, p1_pos[1]:p1_pos[1] + patch_size]
    p2 = img[p2_pos[0]:p2_pos[0] + patch_size, p2_pos[1]:p2_pos[1] + patch_size]
    return p1, p2


def get_patches_similar_by_number_of_foreground_pixels(img: np.ndarray,
                                                       patch_size: int) -> Tuple:
    """
    "Find patch which similarity bigger than  value (so patches are similar - e.g. both centered at text lines)
    :param img: image to take patches from
    :param patch_size: size of patch
    :return: patch 1 and patch 2, with label
    """
    for _ in itertools.count():
        p1, p2 = get_patches(img=img, patch_size=patch_size)
        s = evaluate_s(p1, p2)
        if s >= S_THRESHOLD:  # This might have to be improved
            label = 0
            return p1, p2, label
        else:
            continue


def get_patches_different_by_number_of_foreground_pixels(img: np.ndarray,
                                                         patch_size: int) -> Tuple:
    """
    "Find patch which similarity smaller than  value (so patches are dissimilar - e.g. one at text, second at brake between them)
    :param img: image to take patches from
    :param patch_size: size of patch
    :return: patch 1 and patch 2, with label
    """
    for _ in itertools.count():
        p1, p2 = get_patches(img=img, patch_size=patch_size)
        s = evaluate_s(p1, p2)
        if s < S_THRESHOLD:  # This might have to be improved
            label = 1
            return p1, p2, label
        else:
            continue


def get_patches_different_by_background_area(img: np.ndarray,
                                             patch_size: int) -> Tuple:
    """
    "find two patches which have almost the same number of white pixels (and only them) - so two white patches
    Sometimes it is not so easy, because of the provided document, so there is a timeout provided, after which functions switch to different function
    :param img: image to take patches from
    :param patch_size: size of patch
    :return: patch 1 and patch 2, with label
    """
    margin = 30
    start = time.time()
    for _ in itertools.count():
        if time.time() - start >= TIMEOUT:  # in case there is lack of white space in the image, after long time switch to different function
            p1, p2, label = get_patches_different_by_number_of_foreground_pixels(img, patch_size)
            return p1, p2, label
        p1, p2 = get_patches(img=img, patch_size=patch_size)
        numPixelsPatch = patch_size ** 2
        _, p1_th = cv2.threshold(p1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, p2_th = cv2.threshold(p2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if cv2.countNonZero(p1_th) > numPixelsPatch - margin or cv2.countNonZero(
                p2_th) > numPixelsPatch - margin:
            label = 1
            return p1, p2, label
        else:
            continue


def get_random_pair(images_path, patch_size):
    """Get a random image and random patch (evenly with label 1 or 0) from dataset location
    there are 3 possibilities on patches generation.
    Patches similar by number of foreground pixels (s > S_THRESHOLD)
    Patches different by number of foreground pixels (s < S_THRESHOLD)
    Patches different by background area (nb of white > nb of black pixels)"""
    images = os.listdir(images_path)
    image_name = np.random.choice(images)
    img = cv2.imread(os.path.join(images_path, image_name), 0)
    gen_func = np.random.choice([get_patches_similar_by_number_of_foreground_pixels,
                                 get_patches_similar_by_number_of_foreground_pixels,
                                 get_patches_different_by_number_of_foreground_pixels,
                                 get_patches_different_by_background_area])
    p1, p2, label = gen_func(img, patch_size)

    return p1, p2, label


def make_dirs(path_to_output: str) -> Tuple[str, str]:
    """
    Preparing directories for train and val dataset
    :param path_to_output: path to folde where data will be stored
    :return: train path, val path
    """
    if os.path.isdir(path_to_output) is False:
        os.mkdir(path_to_output)
    train_path = os.path.join(path_to_output, "train")
    val_path = os.path.join(path_to_output, "val")
    if os.path.isdir(train_path) is False:
        os.mkdir(train_path)
    if os.path.isdir(val_path) is False:
        os.mkdir(val_path)
    if os.path.isdir(os.path.join(val_path, "val_0")) is False:
        os.mkdir(os.path.join(val_path, "val_0"))
    if os.path.isdir(os.path.join(val_path, "val_1")) is False:
        os.mkdir(os.path.join(val_path, "val_1"))
    if os.path.isdir(os.path.join(train_path, "train_0")) is False:
        os.mkdir(os.path.join(train_path, "train_0"))
    if os.path.isdir(os.path.join(train_path, "train_1")) is False:
        os.mkdir(os.path.join(train_path, "train_1"))
    return train_path, val_path


if __name__ == "__main__":
    tests.show_generated_patch(
        images_path=os.path.join(os.path.dirname(os.path.abspath(os.getcwd())), 'data','cutted_pages'),
        patch_size=100)
