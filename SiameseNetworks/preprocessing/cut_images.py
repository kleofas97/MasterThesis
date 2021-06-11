import os
import cv2
# HERE PUT NAME OF IAM FOLDER E.G. PracaMagisterka/SiameseNetworks/data/IAM
IMAGES_FOLDER_PATH = "IAM"

PATH_TO_IMAGES = os.path.join(os.path.dirname(os.path.abspath(os.getcwd())), 'data','IAM')
PATH_TO_SAVE = os.path.join(os.path.dirname(os.path.abspath(os.getcwd())), 'data','cutted_pages')

if __name__ == "__main__":

    scale_percent = 25  # percent of original size
    if os.path.isdir(PATH_TO_SAVE) is False:
        os.mkdir(PATH_TO_SAVE)

    for imgs in os.listdir(PATH_TO_IMAGES):
        img = cv2.imread('{}/{}'.format(PATH_TO_IMAGES, imgs), 0)
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img_smaller = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        roi = cv2.selectROI(img_smaller)
        roi_cropped = img[int(roi[1]*4):int((roi[1] + roi[3])*4), int(roi[0]*4):int((roi[0] + roi[2])*4)]
        cv2.imwrite(os.path.join(PATH_TO_SAVE,imgs), roi_cropped)
