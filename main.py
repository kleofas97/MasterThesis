import matlab.engine
import numpy as np
import cv2
from CNN_RNN_CTC.src.Model import Model, DecoderType
from  CNN_RNN_CTC.src.main import FilePaths,infer_matlab
import os

"""
Script which send the sample ID to matlab, where the img is read and segmented. Then, an image with 
labeled lines is sent back to the this script. Then the text is read by the AI 
and predicted text put on original image .

"""

# for text
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (0, 0, 0)
thickness = 2

decoderType = DecoderType.BestPath
eng = matlab.engine.start_matlab()
eng.addpath(os.path.join(os.getcwd(),'Scale-Space Matlab'))
model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True)
for i in range(1, 25):
    test_sample = i
    name = os.path.join(os.getcwd(), 'tests', 'final_test_set', 'input',
                        "atest" + str(test_sample) + ".png")
    img_org = cv2.imread(name, cv2.IMREAD_GRAYSCALE)

    ret = eng.FunExtractLines(test_sample)
    img_mat = np.asarray(ret)
    max_row_val = np.amax(img_mat, axis=1)
    indexes = np.unique(max_row_val, return_index=True)[1]
    lines = [int(max_row_val[index]) for index in sorted(indexes)]
    batch = []
    text_coord = []
    print("Lines extracted")
    for index, line in enumerate(lines):  # loop over next unique label
        if line != 0:  # if not the background
            img = img_mat.copy()
            img[img != line] = 0  # zero all values that are not used in particular value
            img = img.astype('uint8')
            x, y, w, h = cv2.boundingRect(
                img)  # find the bounding rectangle of nonzero points in the image
            img_org_copy = img_org.copy()
            img_org_copy[img != line] = 255
            # crop the image
            cropped_img = img_org_copy[y:y + h, x:x + w]
            batch.append(cropped_img)
            text_coord.append((x, y + h + 20))
    # predict the text on the whole text
    predictions, probas = infer_matlab(model, batch)
    # put text on original img
    for index, (coord, prediction) in enumerate(zip(text_coord, predictions), start=1):
        to_print = prediction.replace("|", " ")
        image = cv2.putText(img_org, to_print, coord, font,
                            fontScale, color, thickness, cv2.LINE_AA)
    # print the result
    scale_percent = 50  # percent of original size
    dim = (int(img_org.shape[1] * scale_percent / 100), int(img_org.shape[0] * scale_percent / 100))
    resized = cv2.resize(img_org, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow("Resized image with text", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
