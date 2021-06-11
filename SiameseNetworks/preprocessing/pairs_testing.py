import os
import numpy as np
import cv2
from typing import List
import matplotlib.pyplot as plt
import SiameseNetworks.preprocessing.my_pairs as my_pairs

"""Module for testing staff in terms of patch generation.
    use:
    'show_generated_patch' for checking how patches are generated
    'test_s_for_images' for checking how for nb of images and different patches s value is generated
"""


def test_patches_generation(img_in: np.ndarray, p1_pos: List, p2_pos: List,
                            patch_size: int) -> None:
    """Test if patches are generated ok in terms of location"""
    img = img_in.copy()
    pkt1 = (p1_pos[1], p1_pos[0])
    pkt2 = (p1_pos[1] + patch_size, p1_pos[0] + patch_size)
    pkt3 = (p2_pos[1], p2_pos[0])
    pkt4 = (p2_pos[1] + patch_size, p2_pos[0] + patch_size)
    cv2.rectangle(img, pkt1, pkt2, (0, 0, 0),thickness=3)
    cv2.rectangle(img, pkt3, pkt4, (0, 0, 0),thickness=3)

    scale_percent = 50  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img_smaller = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    plt.imshow(img_smaller, cmap=plt.cm.gray)
    plt.show()


def test_gotten_s_values(s: List, patch_size: int, nb_of_images: int, save=False) -> None:
    """Ilustrate value of s, provided by the patch comparision"""
    s_arr = np.array(s)
    s_arr *= 100
    plt.hist(s_arr, bins=20, range=(80, 100))
    plt.title("Patch size: {}".format(patch_size))
    if save:
        plt.savefig("s_histogram_for_{}_images_and_patch_{}.jpg".format(nb_of_images, patch_size))
    plt.show()


def test_s_for_images(images_path, nb_of_images, min_patch_size=50, max_patch_size=200, step=20,
                      save=False):
    """Evaluate S value for nb of images, and patches """
    images = os.listdir(images_path)

    for size in range(min_patch_size, max_patch_size, step):
        s_list = []
        for _ in range(nb_of_images):
            image_name = np.random.choice(images)
            img = cv2.imread(os.path.join(images_path, image_name), 0)
            s_list += my_pairs.get_s_list(img, size,
                                          nb_of_patches=10000)  # list of s for current size and image
        test_gotten_s_values(s_list, size, nb_of_images=nb_of_images, save=save)


def show_generated_patch(images_path: str, patch_size: int) -> None:
    """Show if patches are generated correctly"""
    images = os.listdir(images_path)
    image_name = np.random.choice(images)
    img = cv2.imread(os.path.join(images_path, image_name), 0)
    p1_pos, p2_pos = my_pairs.get_position(img, patch_size)
    test_patches_generation(img, p1_pos, p2_pos, patch_size)

if __name__ == "__main__":
    show_generated_patch(images_path=r"F:\Studia\pythonProject\unsupervised_line_segmentation\my_way\data\cutted_pages\part1",patch_size=100)