
import os, random, shutil, glob
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import pandas as pd

###############################################################################
# this script removes 200 images from the adience dataset per group for 0-2,  #
# 4-6, 8-13, and 15-20. it augments these images and places them in the       #
# correct folders for the wiki dataset.                                       #
#                                                                             #
# NOTE:                                                                       #
# run this script BEFORE running populate_subfolders.py script (delete        #
# existing folders if needed). then run populate_subfolders.py                #
###############################################################################


def load_all_folds_info(raw_path, age_groups):
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
        # Faces file:
        fold = np.genfromtxt(
            os.path.join(raw_path, "fold_%d_data.txt" % j)
            , usecols=(0, 1, 2, 3, 4)
            , delimiter='\t'
            , dtype=None
        )[1:]
        base_dir_column = np.array([os.path.join(raw_path, "faces/faces/")] * fold.shape[0]).reshape((fold.shape[0], 1))
        fold_num_column = np.array([j] * fold.shape[0]).reshape((fold.shape[0], 1))
        fold = np.hstack((fold, base_dir_column, fold_num_column))
        folds.append(fold)

        # Aligned file:
        fold = np.genfromtxt(
            os.path.join(raw_path, "fold_frontal_%d_data.txt" % j)
            , usecols=(0, 1, 2, 3, 4)
            , delimiter='\t'
            , dtype=None
        )[1:]
        base_dir_column = np.array([os.path.join(raw_path, "aligned/aligned/")] * fold.shape[0]).reshape(
            (fold.shape[0], 1))
        fold_num_column = np.array([j] * fold.shape[0]).reshape((fold.shape[0], 1))
        # print("shapes: ", fold.shape, base_dir_column.shape, fold_num_column.shape)
        fold = np.hstack((fold, base_dir_column, fold_num_column))
        folds.append(fold)
    adience_images = np.vstack(folds)
    # print("adience_images.shape: ", adience_images.shape)
    adience_images = add_img_counts(adience_images)

    # Filter some data:
    before = adience_images.shape[0]
    keep = [False]*adience_images.shape[0]
    for idx, img in enumerate(adience_images):
        age = get_age(img)
        if age is not None and age in age_groups.keys():
            keep[idx] = True
            adience_images[idx,3] = img[3].replace("(8, 12)", "(8, 13)")
    adience_images = adience_images[keep]
    after = adience_images.shape[0]
    # print("Removed {} images on age filtering".format(after-before))

    col_names = ["user_id", "filename", "face_id", "age", "gender", "base_path", "fold_num", "img_count"]

    adience_img_count = adience_images.shape[0]
    return adience_images, adience_img_count, col_names


def add_img_counts(adience_images):
    counts = []
    for img in adience_images:
        counts.append(len(get_raw_paths(img)))
    counts_col = np.array(counts).reshape((adience_images.shape[0], 1))
    return np.hstack((adience_images, counts_col))

##
## Accessors for adience_images np array:
def get_userid(img):
    return img[0]


def get_filename(img):
    return img[1]


def get_face_id(img):
    return img[2]


def get_age(img):
    age = img[3]
    if age is not None and len(age) > 5:
        age = age[1:-1].replace(', ', '-')
        return age.replace("8-12", "8-13")
    return None


def get_gender(img):
    gender = img[4]
    return gender if len(gender) > 0 else None


def get_raw_paths(img):
    img_path = img[5]
    img_path = os.path.join(img_path, get_userid(img))
    img_paths = glob.glob("{}/*.{}.*{}".format(img_path, get_face_id(img), get_filename(img)))
    return img_paths


def get_fold_num(img):
    return img[6]


def get_img_count(img):
    return img[7]


def sample_adience_images(adience_images, num_samples, age_group):
    """
    Picks which images to move from adience to wiki/imdb. Can't guarantee exactly num_samples,
    the guarantee is only that at least num_samples images will be pulled from adience for the
    specified age_group.

    This is because if we pull one user's (face_id) img, we have to pull all of that user's img's, so we
    avoid LOPO ("Leave One Person Out" issues).

    :param adience_images: 2d np array of img info from adience .txt files
    :param adience_indexes: list of indexes into adience_images that haven't been sampled yet
    :param num_samples: # samples to take
    :return: list of sampled images. Each item in list is a dictionary. key/val map:
                ["paths"]: list of paths for sampled image
                ["img"]: row from adience_images np array corresponding to the sampled image
    """

    age_low, age_hi = age_group.split("-")
    age_low, age_hi = int(age_low), int(age_hi)
    age_bucket = "({}, {})".format(age_low, age_hi)
    adience_agegroup = adience_images[adience_images[:,3]==age_bucket]
    print("Total images for age_group {}: {}".format(age_group, adience_agegroup.shape[0]))

    # All face_id's for the age_group:
    face_ids = np.unique(adience_agegroup[:,2])
    np.random.shuffle(face_ids)
    # print("face_ids: ", face_ids)
    # print("face_ids.shape: ", face_ids.shape)

    # Gather paths of images we will augment and move:
    sampled_images = []  # images we will augment and move
    sampled_count = 0
    sampled_face_ids = []

    for face_id in face_ids:
        # Grab all imgs for curr. face_id:
        imgs = adience_agegroup[adience_agegroup[:,2]==face_id]
        print("face_id: {}, imgs: {}".format(face_id, imgs.shape))

        if sampled_count >= num_samples:
            break
        sampled_face_ids.append(face_id)
        for img in imgs:
            paths = get_raw_paths(img)
            sampled_images.append({"paths": paths, "img": img })
            sampled_count += len(paths)

    print("age_group:'{}', #sampled_imgs: {}, #sampled_images: {}".format(age_group, sampled_count,
                                                                           len(sampled_images)))
    print("#face_id's sampled: {}".format(len(np.unique(sampled_face_ids))))
    return sampled_images


def augment_img(datagen, target_dir, img, counts, train_or_valid):
    """

    :param datagen:
    :param target_dir:
    :param x: image to augment
    :param counts: Dictionary of counts per directory, of # augmentations generated
    :return:
    """
    NUM_AUG = 15
    aug_count = 0
    # print("\taugmenting img target path: ", target_dir)

    # print("target_dir: ", target_dir)
    train_dir = target_dir\
        .replace("age_gender/", "age_gender/train/")\
        .replace("/gender/", "/gender/train/")\
        .replace("/age/", "/age/train/")
    val_dir = target_dir\
        .replace("age_gender/", "age_gender/valid/")\
        .replace("/gender/", "/gender/valid/")\
        .replace("/age/", "/age/valid/")

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    # print("train_dir: ", train_dir)
    # print("val_dir: ", val_dir)
    target_dir = train_dir if train_or_valid == "train" else val_dir
    # print("target_dir: ", target_dir)

    if target_dir not in counts:
        counts[target_dir] = 0

    for augmented in datagen.flow(img, batch_size=1, save_to_dir=target_dir, save_prefix='aug', save_format='jpeg'):
        if aug_count >= NUM_AUG:
            break
        aug_count += 1
        counts[target_dir] += 1


def augment_and_move(
    sampled_images, datagen, counts
    , male_dir, female_dir, age_dir, age_gender_dir
):
    """
    :param adience_images: 2d np array of image data from the adience .txt files
    :param sampled_indexes: indxes (into adience_images) to augment & move
    :param sampled_images: tuple
                            [0]: list of paths corresponding to the sampled img
                            [1]: row from adience_images corresponding to the sampled image
    :return:
    """

    VALID_RATIO = 10
    num_samples = sum([len(s["paths"]) for s in sampled_images])
    valid_start_index =  num_samples//VALID_RATIO
    print("num_samples, valid_ratio: ", num_samples, num_samples//VALID_RATIO)
    num_train, num_valid = 0, 0
    num_deleted = 0
    processed = []
    i = 0
    for sample in sampled_images:
        img_paths, img_data = sample["paths"], sample["img"]
        for img_path in img_paths:
            # print("augmenting img: ", img_path)
            img = load_img(img_path)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            gender, age = get_gender(img_data), get_age(img_data)

            # Is this img in the train or val split?
            if i < valid_start_index:
                train_or_valid = "valid"
                num_valid += 1
            else:
                train_or_valid = "train"
                num_train += 1
            # print("train_or_valid: ", train_or_valid)

            if gender=="m":
                augment_img(datagen, male_dir, x, counts, train_or_valid)
                augment_img(datagen, age_dir + age, x, counts, train_or_valid)
                augment_img(datagen, age_gender_dir + 'male_' + age, x, counts, train_or_valid)
            elif gender=="f":
                augment_img(datagen, female_dir, x, counts, train_or_valid)
                augment_img(datagen, age_dir + age, x, counts, train_or_valid)
                augment_img(datagen, age_gender_dir + 'female_' + age, x, counts, train_or_valid)
            else:
                augment_img(datagen, age_dir + age, x, counts, train_or_valid)
            processed.append(img_path)
            num_deleted += 1
            i += 1

    print("num_train, num_valid: ", num_train, num_valid)
    print("Deleting {} imgs from adience raw".format(len(processed)))
    for img in processed:
        if os.path.exists(img) and img.endswith(".jpg"):
            os.remove(img)
            print("Deleting ", img)
            pass
        else:
            print("Not removing: ", img)

    return num_deleted


def to_df(arr, cols):
    data = {col_name:arr[:,i] for i,col_name in enumerate(cols)}
    df = pd.DataFrame(data)
    return df


def explore_data():
    root = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(root, "raw")
    age_groups = {"0-2":0, "4-6":0, "8-13":0, "15-20":0}
    adience_images, adience_img_count, col_names = load_all_folds_info(raw_path, age_groups)
    df = to_df(adience_images, col_names)

    df = df[df['age'].isin(
        ["(0, 2)", "(4, 6)", "(8, 12)", "(15, 20)", "(25, 32)", "(38, 43)", "(48, 53)", "(60, 100)"]
    )]
    print(df.sort_values(by=["age", "fold_num", "face_id"]).groupby(["age", "fold_num", "face_id"]).img_count.sum())


def main():
    np.random.seed(57)
    datagen = ImageDataGenerator(
        rotation_range=15.0,
        brightness_range=[0.1,1.5],
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest',
        vertical_flip=False,
        horizontal_flip=True
    )
    root = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(root, "raw")
    target_path = os.path.join(root, "processed/wiki")

    gender, age, age_gender = target_path + '/gender', target_path + '/age', target_path + '/age_gender'
    paths = [gender, age, age_gender] + [gender + '/female', gender + '/male']
    paths += [age + p for p in ['/a_0-2', '/b_4-6', '/c_8-13', '/d_15-20', '/e_25-32', '/f_38-43', '/g_48-53',
                                '/h_60-130']]
    paths += [age_gender + p for p in ['/i_male_0-2', '/j_male_4-6', '/k_male_8-13', '/l_male_15-20', '/m_male_25-32',
                                       '/n_male_38-43', '/o_male_48-53', '/p_male_60-130']]
    paths += [age_gender + p for p in ['/a_female_0-2', '/b_female_4-6', '/c_female_8-13', '/d_female_15-20',
                                       '/e_female_25-32', '/f_female_38-43', '/g_female_48-53', '/h_female_60-130']]

    male, female, age_path = target_path + '/gender/male', target_path + '/gender/female', target_path + '/age/'
    print("processed_path: ", target_path)
    print("male, female, age_path : ", male, female, age_path )

    age_groups = {"0-2":0, "4-6":0, "8-13":0, "15-20":0}
    # how many samples to move from adience, for each age_group:
    NUM_SAMPLES = 200

    # Load all the adience image info into a single numpy matrix:
    adience_images, adience_img_count, col_names = load_all_folds_info(raw_path, age_groups)


    # For each age_group, sample images from adience, and move to wiki:
    counts = {}
    num_deleted = 0
    for age_group, age_group_count in age_groups.items():
        print("Sampling from adience for age_group: ", age_group)
        sampled_images = sample_adience_images(adience_images, NUM_SAMPLES, age_group)
        num_deleted_tmp = augment_and_move(
            sampled_images
            , datagen
            , counts
            , male, female, age_path, age_gender + "/"
        )
        num_deleted += num_deleted_tmp

        print("Finished augment&move for age_group:'{}'. Deleted from adience: {}".format(age_group, num_deleted_tmp))
        print("Num images sampled: ", sum([len(s["paths"]) for s in sampled_images]))
        print()

    print("augmented counts: ", counts)
    print("Total Deleted from adience: ", num_deleted)
    print("Done! ")





## =====================================================================================================================

if __name__ == '__main__':
    main()
    # explore_data()
