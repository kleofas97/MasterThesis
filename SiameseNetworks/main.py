import SiameseNetworks.model as model_op
import tensorflow
import os
import numpy as np
import SiameseNetworks.preprocessing.my_pairs as pairs
from SiameseNetworks import Arguments,predict

"""Before running main.py - put data into the data directory, as there are only few examples for now. 
Original IAM Dataset needs to be cropped to contain only text, without the printed text etc.
To cut the data use cut_images.py
Then the Dataset need to be splitted, so run split_dataset.py
Then run this script.
"""



Args = Arguments.parse_args()
if Args.prepare_dataset:
    pairs.prepare_dataset_on_disk(dataset_path_train=os.path.join(Args.dataset_path, "train"),
                                  dataset_path_val=os.path.join(Args.dataset_path, "val"),
                                  path_to_output=Args.dataset_prep_dir,
                                  train_set_size=Args.train_set_size,
                                  val_set_size=Args.validation_set_size,
                                  patch_size=Args.input_shape)
nb_of_samples_train = len(os.listdir(os.path.join(Args.dataset_prep_dir, 'train', 'train_0')))
nb_of_samples_val = len(os.path.join(Args.dataset_prep_dir, 'val', 'val_0'))
train_steps = np.floor(nb_of_samples_train / Args.batch_size)
val_steps = np.floor(nb_of_samples_val / Args.batch_size)
input_shape = (Args.input_shape, Args.input_shape, 1)
print(nb_of_samples_train)
if Args.train == False:
    # prepare generators
    generator_train = model_op.genereate_batch(
        path_1=os.path.join(Args.dataset_prep_dir, 'train', 'train_0'),
        path_2=os.path.join(Args.dataset_prep_dir, 'train', 'train_1'),
        batch_size=Args.batch_size)
    generator_val = model_op.genereate_batch(
        path_1=os.path.join(Args.dataset_prep_dir, 'val', 'val_0'),
        path_2=os.path.join(Args.dataset_prep_dir, 'val', 'val_1'),
        batch_size=Args.batch_size)
    if Args.continue_from_best is True:
        assert Args.path_to_model is not None, "invalid path to model"
        model = tensorflow.keras.models.load_model(Args.path_to_model)
    else:
        model = model_op.built_model(input_shape=input_shape)
    model, history = model_op.fit_model(model=model, generator_train=generator_train,
                                        path_to_model=Args.path_to_model,
                                        generator_val=generator_val,
                                        epochs=Args.epochs,
                                        learning_rate=Args.learning_rate, train_steps=train_steps,
                                        val_steps=val_steps, patch_size=Args.input_shape)
else:
    # model = tensorflow.keras.models.load_model(os.path.join(Args.path_to_model,'150px_bestmodel.h5py'))
    # prediction = model.predict(Args.test_img_path)
    predict.pca(path_to_model=os.path.join(Args.path_to_model, str(Args.input_shape) + 'px_bestmodel.h5py'),test_folder=Args.test_img_path)

