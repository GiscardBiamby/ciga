import os, random, shutil, glob
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import pandas as pd

def get_gender(img):
    gender = img[4]
    return gender if len(gender) > 0 else None

def get_age(img):
    age = img[3]
    if age is not None and len(age) > 5:
        age = age[1:-1].replace(', ', '-')
        return age.replace("8-12", "8-13")
    return None

def get_fold_num(img):
    return img[6]

def get_filename(img):
    return img[1]

def get_userid(img):
    return img[0]

def get_raw_path(img):
    img_path = img[5]
    img_path = os.path.join(img_path, get_userid(img))
    img_paths = glob.glob("{}/*{}".format(img_path, img[1]))
    return img_paths

def load_all_folds_info(raw_path):
    """
    Return 2d numpy array of all columns 1-5 from all the .txt files of the adience dataset. ALso appends one new
    column which is the base path that points to local disk path depending on which block of data the image is from.
        if image from "faces": columnn6 = /home/.../ciga/datasets/raw/faces/faces/'
        if image from "aligned": columnn6 = '/home/.../ciga/datasets/raw/aligned/aligned/'
    :param raw_path:
    :return: tuple:
                adience_images: 2d numpy array
                adience_img_count: # images in adience_images
    """
    folds = []
    for j in range(5):
        fold = np.genfromtxt(os.path.join(raw_path, "fold_%d_data.txt" % j), usecols=(0, 1, 2, 3, 4),
                             delimiter='\t', dtype=None)

        base_dir_column = np.array([os.path.join(raw_path, "faces/faces/")] * fold.shape[0]).reshape((fold.shape[0], 1))
        fold_num_column = np.array([j] * fold.shape[0]).reshape((fold.shape[0], 1))
        fold = np.hstack((fold, base_dir_column, fold_num_column))
        folds.append(fold)

        fold = np.genfromtxt(os.path.join(raw_path, "fold_frontal_%d_data.txt" % j), usecols=(0, 1, 2, 3, 4),
                             delimiter='\t',
                             dtype=None)
        base_dir_column = np.array([os.path.join(raw_path, "aligned/aligned/")] * fold.shape[0]).reshape(
            (fold.shape[0], 1))
        fold_num_column = np.array([j] * fold.shape[0]).reshape((fold.shape[0], 1))
        fold = np.hstack((fold, base_dir_column, fold_num_column))
        folds.append(fold)
    adience_images = np.vstack(folds)
    # print("adience_images.shape: ", adience_images.shape)

    # Filter some data:
    # before = adience_images.shape[0]
    # keep = [False]*adience_images.shape[0]
    # for idx, img in enumerate(adience_images):
    #     age = get_age(img)
    #     if age is not None and age in age_groups.keys():
    #         keep[idx] = True
    # adience_images = adience_images[keep]
    # after = adience_images.shape[0]
    # print("Removed {} images on age filtering".format(after-before))


    adience_img_count = adience_images.shape[0]
    return adience_images, adience_img_count

def generateFolds(adience_images, processed_path, fold_num):
    """
    Fold i goes into val, and others go into train
    :param adience_images:
    :param processed_path:
    :param fold_num:
    :return:
    """
    age_group_map = {
        "0-2": "a_0-2/", "4-6": "b_4-6/", "8-13": "c_8-13/", "15-20": "d_15-20/", "25-32": "e_25-32/",
        "38-43": "f_38-43/", "48-53": "g_48-53/", "60-100": "h_60-130/"
    }
    gender_map = {
        "m": "male/", "f": "female/"
    }
    age_gender_map = {
        ("0-2", "f"): "a_female_0-2/"
        , ("4-6", "f"): "b_female_4-6/"
        , ("8-13", "f"): "c_female_8-13/"
        , ("15-20", "f"): "d_female_15-20/"
        , ("25-32", "f"): "e_female_25-32/"
        , ("38-43", "f"): "f_female_38-43/"
        , ("48-53", "f"): "g_female_48-53/"
        , ("60-100", "f"): "h_female_60-130/"
        , ("0-2", "m"): "i_male_0-2/"
        , ("4-6", "m"): "j_male_4-6/"
        , ("8-13", "m"): "k_male_8-13/"
        , ("15-20", "m"): "l_male_15-20/"
        , ("25-32", "m"): "m_male_25-32/"
        , ("38-43", "m"): "n_male_38-43/"
        , ("48-53", "m"): "o_male_48-53/"
        , ("60-100", "m"): "p_male_60-130/"
    }
    gender_base_path = os.path.join(processed_path, "gender_fold_{}/train_or_val/".format(fold_num))
    age_base_path = os.path.join(processed_path, "age_fold_{}/train_or_val/".format(fold_num))
    age_gender_base_path = os.path.join(processed_path, "age_gender_fold_{}/train_or_val/".format(fold_num))

    for img in adience_images:
        gender = get_gender(img)
        age = get_age(img)
        img_paths = get_raw_path(img)
        img_fold = str(get_fold_num(img))
        split = "valid" if  img_fold==str(fold_num) else "train"
        # print("fold_num: {}, imgfold: {}, split: {}".format(type(fold_num), type(img_fold), split))

        for img_path in img_paths:
            # print("from: ", img_path)
            # # Gender:
            if gender is not None and gender in gender_map:
                to_dir = os.path.join(gender_base_path, gender_map[gender]).replace("train_or_val", split)
                to_path = os.path.join(to_dir, os.path.basename(img_path))
                if not os.path.exists(to_dir):
                    os.makedirs(os.path.dirname(to_dir))
                if not os.path.exists(to_path):
                    shutil.copy(img_path, to_path)
                # print("to: ", to_path)
            else:
                # print("Not copying img b/c of unwanted or invalid gender: ", gender)
                pass

            # # Age:
            if age is not None and age in age_group_map:
                to_dir = os.path.join(age_base_path, age_group_map[age]).replace("train_or_val", split)
                to_path = os.path.join(to_dir, os.path.basename(img_path))
                if not os.path.exists(to_dir):
                    os.makedirs(os.path.dirname(to_dir))
                if not os.path.exists(to_path):
                    shutil.copy(img_path, to_path)
                # print("to: ", to_path)
            else:
                # print("Not copying img b/c of unwanted or invalid age: ", age)
                pass

            # Age-Gender:
            if (gender is not None and gender in gender_map) and (age is not None and age in age_group_map):
                to_dir = os.path.join(age_gender_base_path, age_gender_map[(age,gender)]).replace("train_or_val", split)
                to_path = os.path.join(to_dir, os.path.basename(img_path))
                if not os.path.exists(to_dir):
                    os.makedirs(os.path.dirname(to_dir))
                if not os.path.exists(to_path):
                    shutil.copy(img_path, to_path)
                # print("to: ", to_path)
            else:
                # print("Not copying img b/c of unwanted or invalid age_gender: ", (age,gender))
                pass

def main():
    raw_path = "./raw/"
    processed_path = "./processed_test/adience/"
    adience_images, adience_img_count = load_all_folds_info(raw_path)
    print(adience_images)
    K = 5
    for i in range(K):
        print("Generating fold {} of {}".format(i+1, K))
        generateFolds(adience_images, processed_path, i)


main()
