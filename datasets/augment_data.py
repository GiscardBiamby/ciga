
import os, random, shutil, glob
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

###############################################################################
# this script removes 200 images from the adience dataset per group for 0-2,  #
# 4-6, 8-13, and 15-20. it augments these images and places them in the       #
# correct folders for the wiki dataset.                                       #
#                                                                             #
# NOTE:                                                                       #
# run this script BEFORE running populate_subfolders.py script (delete        #
# existing folders if needed). then run populate_subfolders.py                #
###############################################################################

num_aug = 15
datagen = ImageDataGenerator(rotation_range=15.0,
    brightness_range=[0.1,1.5],
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest',
    vertical_flip=False,
    horizontal_flip=True)

root = os.path.dirname(os.path.abspath(__file__))
raw_path = os.path.join(root, "raw")
processed_path = os.path.join(root, "processed/wiki")

gender, age, age_gender = processed_path + '/gender', processed_path + '/age', processed_path + '/age_gender'
paths = [gender, age, age_gender] + [gender + '/female', gender + '/male']
paths += [age + p for p in ['/0-2', '/4-6', '/8-12', '/15-20', '/25-32', '/38-43', '/48-53', '/60-100']]
paths += [age_gender + p for p in ['/male_0-2', '/male_4-6', '/male_8-12', '/male_15-20', '/male_25-32', '/male_38-43', '/male_48-53', '/male_60-100']]
paths += [age_gender + p for p in ['/female_0-2', '/female_4-6', '/female_8-12', '/female_15-20', '/female_25-32', '/female_38-43', '/female_48-53', '/female_60-100']]

for p in paths:
    if not os.path.exists(p):
        os.makedirs(p)

male, female, age_path = processed_path + '/gender/male', processed_path + '/gender/female', processed_path + '/age/'
age_gender += '/'

age_groups = {"0-2":0, "4-6":0, "8-13":0, "15-20":0}
stop = False

# iterate over the adience folds
for j in range(5):
    if stop:
        break

    folds = [np.genfromtxt(os.path.join(raw_path, "fold_%d_data.txt" % j), usecols=(0,1,2,3,4), delimiter='\t', dtype=None)]
    folds += [np.genfromtxt(os.path.join(raw_path, "fold_frontal_%d_data.txt" % j), usecols=(0,1,2,3,4), delimiter='\t', dtype=None)]
    for fold in folds:
        if stop:
            break

        for i in range(1, fold.shape[0]):
            if stop:
                break

            age = fold[i, 3].decode('UTF-8')
            if age is not None and len(age) > 5:
                age = age[1:-1].replace(', ', '-')
                if age not in age_groups.keys():
                    break
                elif age_groups[age] >= 200:
                    break
            else:
                break

            gender = fold[i, 4].decode('UTF-8')
            gender = gender if len(gender) > 0 else None

            path, img, id = fold[i, 0].decode('UTF-8'), fold[i, 1].decode('UTF-8'), fold[i, 2].decode('UTF-8')
            face_path = raw_path + '/faces/faces/' + path
            aligned_path = raw_path + '/aligned/aligned/' + path
            paths = list(glob.iglob(os.path.join(face_path, "*" + id + "." + img)))
            paths += list(glob.iglob(os.path.join(aligned_path, "*" + id + "." + img)))

            if gender == 'm':
                for src in paths:
                    img = load_img(src) 
                    x = img_to_array(img)
                    x = x.reshape((1,) + x.shape)

                    n = 0
                    for batch in datagen.flow(x, batch_size=1, save_to_dir=male, save_prefix='aug', save_format='jpeg'):
                        n += 1
                        if n > num_aug:
                            break 

                    n = 0
                    for batch in datagen.flow(x, batch_size=1, save_to_dir=age_path + age, save_prefix='aug', save_format='jpeg'):
                        n += 1
                        if n > num_aug:
                            break 

                    n = 0
                    for batch in datagen.flow(x, batch_size=1, save_to_dir=age_gender + 'male_' + age, save_prefix='aug', save_format='jpeg'):
                        n += 1
                        if n > num_aug:
                            break 

            elif gender == 'f':
                for src in paths:
                    img = load_img(src) 
                    x = img_to_array(img)
                    x = x.reshape((1,) + x.shape)

                    n = 0
                    for batch in datagen.flow(x, batch_size=1, save_to_dir=female, save_prefix='aug', save_format='jpeg'):
                        n += 1
                        if n > num_aug:
                            break 

                    n = 0
                    for batch in datagen.flow(x, batch_size=1, save_to_dir=age_path + age, save_prefix='aug', save_format='jpeg'):
                        n += 1
                        if n > num_aug:
                            break 

                    n = 0
                    for batch in datagen.flow(x, batch_size=1, save_to_dir=age_gender + 'female_' + age, save_prefix='aug', save_format='jpeg'):
                        n += 1
                        if n > num_aug:
                            break 

            else:
                for src in paths:
                    img = load_img(src) 
                    x = img_to_array(img)
                    x = x.reshape((1,) + x.shape)

                    n = 0
                    for batch in datagen.flow(x, batch_size=1, save_to_dir=age_path + age, save_prefix='aug', save_format='jpeg'):
                        n += 1
                        if n > num_aug:
                            break 

            age_groups[age] += 1
            stop = True
            for group in age_groups:
                if age_groups[group] < 200:
                    stop = False

            os.remove(src)




