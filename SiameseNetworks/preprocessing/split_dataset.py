import numpy as np
import os
import shutil

def splitImages(path_to_dataset: str, path_to_output:str,train_size = 0.8) -> None:
    if os.path.isdir(path_to_output) is False:
        os.mkdir(path_to_output)
    if os.path.isdir(os.path.join(path_to_output,"train")) is False:
        os.mkdir(os.path.join(path_to_output,"train"))
    if os.path.isdir(os.path.join(path_to_output,"val")) is False:
        os.mkdir(os.path.join(path_to_output,"val"))
    img_list = os.listdir(path_to_dataset)
    for img in img_list:
        if(np.random.choice((0,1),p=[train_size,1-train_size])):
            #if 1 - val
            shutil.copyfile(os.path.join(path_to_dataset,img), os.path.join(path_to_output,"val",img))
        else:
            shutil.copyfile(os.path.join(path_to_dataset, img),
                            os.path.join(path_to_output, "train", img))

PATH_TO_DATASET = os.path.join(os.path.dirname(os.path.abspath(os.getcwd())), 'data','cutted_pages')
PATH_TO_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(os.getcwd())), 'data', 'data_split')
splitImages(path_to_dataset=PATH_TO_DATASET,path_to_output=PATH_TO_OUTPUT)
