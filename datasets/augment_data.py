
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


# If true, script won't modify/copy/create/move any files, just output what it *would* do if TEST_MODE were disabled:
TEST_MODE = False


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

    # Scrub some data:
    for idx, img in enumerate(adience_images):
        age = get_age(img)
        if age is not None and age in age_groups.keys():
            # keep[idx] = True
            adience_images[idx,3] = img[3].replace("(8, 12)", "(8, 13)")

    col_names = ["user_id", "filename", "face_id", "age", "gender", "base_path", "fold_num"]

    # Only need this during data exploration, it makes the script slower so disable it if not needed:
    if False:
        adience_images = add_img_counts(adience_images)
        col_names += "img_count"

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


def get_all_paths_for_face_id(face_id):
    img_paths = set()
    img_paths.update(set(glob.glob("./raw/faces/faces/*/*.{}.*.jpg".format(face_id))))
    img_paths.update(set(glob.glob("./raw/aligned/aligned/*/*.{}.*.jpg".format(face_id))))
    # print("img_paths: ", list(img_paths))
    return list(img_paths)


def sample_adience_images(adience_images, col_names, num_samples, age_group, alredy_sampled):
    """
    Picks which images to move from adience to wiki/imdb. Can't guarantee exactly num_samples,
    the guarantee is only that at least num_samples images will be pulled from adience for the
    specified age_group.

    This is because if we pull one user's (face_id) img, we have to pull all of that user's img's, so we
    avoid LOPO ("Leave One Person Out" issues).

    :param adience_images: 2d np array of img info from adience .txt files
    :param adience_indexes: list of indexes into adience_images that haven't been sampled yet
    :param num_samples: # samples to take
    :return: dictionary of sampled faces. key/val map:
                key: (string) face_id
                val: tuple, length 3, of info for the face_id: ((str) age, (str) gender, (list) image_paths)
    """

    ignored_faces = get_ignored_faces(adience_images, col_names) # don't sample these face_ids
    age_low, age_hi = age_group.split("-")
    age_low, age_hi = int(age_low), int(age_hi)
    age_bucket = "({}, {})".format(age_low, age_hi)
    adience_agegroup = adience_images[adience_images[:,3]==age_bucket]
    print("Total images for age_group {}: {}".format(age_group, adience_agegroup.shape[0]))

    # All face_id's for the age_group:
    face_ids = np.unique(adience_agegroup[:,2])
    face_ids = [face_id for face_id in face_ids if face_id not in ignored_faces]
    face_ids = [face_id for face_id in face_ids if face_id not in alredy_sampled]
    np.random.shuffle(face_ids)
    # print("face_ids: ", face_ids)
    # print("face_ids.shape: ", face_ids.shape)

    # Gather paths of images we will augment and move:
    sampled_img_count = 0
    sampled_faces = {}

    for face_id in face_ids:
        if sampled_img_count >= num_samples:
            break
        img = adience_agegroup[adience_agegroup[:,2]==face_id][0]
        paths = get_all_paths_for_face_id(face_id)
        print("sampled_img_count: {}, face_id: {}, num_imgs: {}".format(sampled_img_count, face_id, len(paths)))
        sampled_faces[face_id] =  (get_age(img), get_gender(img), paths)
        sampled_img_count += len(paths)

    print("age_group:'{}', #sampled_img_count: {}".format(age_group, sampled_img_count))
    print("#face_id's sampled: {}".format(len(sampled_faces.keys())))
    return sampled_faces


def augment_img(datagen, target_dir, img, counts, train_or_valid, curr_num, total):
    """
    Uses keras ImageDataGenerator to create 15 augmented/transformed versions of the given image.
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
    print("Adding augmented images ({}/{}) to path: {}".format(curr_num, total, target_dir))

    if target_dir not in counts:
        counts[target_dir] = 0

    if not TEST_MODE:
        for augmented in datagen.flow(img, batch_size=1, save_to_dir=target_dir, save_prefix='aug', save_format='jpeg'):
            if aug_count >= NUM_AUG:
                break
            aug_count += 1
            counts[target_dir] += 1


def augment_and_move(
    sampled_images, datagen, counts
    , gender_dir, age_dir, age_gender_dir
    , train_or_valid
):
    """
    :param adience_images: 2d np array of image data from the adience .txt files
    :param sampled_indexes: indxes (into adience_images) to augment & move
    :param sampled_images: tuple
                            [0]: list of paths corresponding to the sampled img
                            [1]: row from adience_images corresponding to the sampled image
    :return:
    """

    # Map age/gender fields from .txt files to folder names based on ground truth labels:
    age_map = {
        "0-2": "a_0-2", "4-6": "b_4-6", "8-13": "c_8-13", "15-20": "d_15-20", "25-32": "e_25-32",
        "38-43": "f_38-43", "48-53": "g_48-53", "60-100": "h_60-130"
    }
    gender_map = {
        "m": "male", "f": "female"
    }
    age_gender_map = {
        ("0-2", "f"): "a_female_0-2"
        , ("4-6", "f"): "b_female_4-6"
        , ("8-13", "f"): "c_female_8-13"
        , ("15-20", "f"): "d_female_15-20"
        , ("25-32", "f"): "e_female_25-32"
        , ("38-43", "f"): "f_female_38-43"
        , ("48-53", "f"): "g_female_48-53"
        , ("60-100", "f"): "h_female_60-130"
        , ("0-2", "m"): "i_male_0-2"
        , ("4-6", "m"): "j_male_4-6"
        , ("8-13", "m"): "k_male_8-13"
        , ("15-20", "m"): "l_male_15-20"
        , ("25-32", "m"): "m_male_25-32"
        , ("38-43", "m"): "n_male_38-43"
        , ("48-53", "m"): "o_male_48-53"
        , ("60-100", "m"): "p_male_60-130"
    }

    num_deleted = 0
    processed = []
    i, total = 1, sum([len(v[2]) for v in sampled_images.values()])
    for face_id, vals in sampled_images.items():
        age, gender, img_paths = vals
        for img_path in img_paths:
            # print("augmenting img: ", img_path)
            img = load_img(img_path)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            age_gender = (age,gender)

            if gender in gender_map:
                augment_img(
                    datagen
                    , os.path.join(gender_dir, gender_map[gender])
                    , x, counts, train_or_valid, i, total
                )
            if age in age_map:
                augment_img(
                    datagen
                    , os.path.join(age_dir, age_map[age])
                    , x, counts, train_or_valid, i, total
                )
            if age_gender in age_gender_map:
                augment_img(
                    datagen
                    , os.path.join(age_gender_dir, age_gender_map[age_gender])
                    , x, counts, train_or_valid, i, total
                )

            processed.append(img_path)
            num_deleted += 1
            i += 1

    print("Deleting {} imgs from adience raw".format(len(processed)))
    for img in processed:
        if os.path.exists(img) and img.endswith(".jpg"):
            if not TEST_MODE:
                os.remove(img)
            print("Deleting (TEST_MODE={}): {}".format(TEST_MODE, img))
            pass
        else:
            print("Not removing: ", img)

    return num_deleted


def get_ignored_faces(adience_images, col_names):
    # Ignore faces that are in 2 folds, seems to happen because we combined faces/faces/ and aligned/aligned/
    df = to_df(adience_images, col_names)
    by_face_fold = df.groupby(["face_id"])["fold_num"].nunique().reset_index(name="freq")
    ignored_face_ids = by_face_fold[by_face_fold.freq==2].face_id.unique()
    # print("ignored: ")
    # print(df[df.face_id.isin(ignored_face_ids)])
    return set(ignored_face_ids)


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

    # df = df[df['age'].isin(
    #     ["(0, 2)", "(4, 6)", "(8, 12)", "(15, 20)", "(25, 32)", "(38, 43)", "(48, 53)", "(60, 100)"]
    # )]
    # print(df.sort_values(by=["age", "fold_num", "face_id"]).groupby(["age", "fold_num", "face_id"]).img_count.sum())

    # See if any face_id's appear in more than one fold:
    print("grouped by: (face_id, fold_num), sorted by: (group size desc). ")
    print(
        df.sort_values(by=["face_id", "fold_num"]).groupby(["face_id", "fold_num"]).size().reset_index(
        name="Freq").sort_values(by='Freq',ascending=False)
    )
    # Results are good, no user shows up in more than one fold!

    # See if any face_id's have more than one fold (should be false):
    print("grouped by: (face_id, fold), sorted by: (group size desc). ")
    print(
        df.groupby(["face_id"])["fold_num"].nunique().reset_index(
        name="Freq").sort_values(by='Freq', ascending=False)
    )

    # See if any face_id's have more than one age (should be false):
    print("grouped by: (face_id, age), sorted by: (group size desc). ")
    print(
        df.groupby(["face_id"])["age"].nunique().reset_index(
        name="Freq").sort_values(by='Freq', ascending=False)
    )

    # See if any face_id's have more than one gender (should be false):
    print("grouped by: (face_id, gender), sorted by: (group size desc). ")
    print(
        df.groupby(["face_id"])["gender"].nunique().reset_index(
        name="Freq").sort_values(by='Freq', ascending=False)
    )

    # See if any face_id's have more than one file (should be true, just a sanity check):
    print("grouped by: (face_id, filename), sorted by: (group size desc). ")
    print(
        df.groupby(["face_id"])["filename"].nunique().reset_index(
        name="Freq").sort_values(by='Freq', ascending=False)
    )

    print(df[df["face_id"]=="436"])

    by_face_fold = df.groupby(["face_id"])["fold_num"].nunique().reset_index(name="freq")
    print(by_face_fold[by_face_fold.freq==2].face_id.unique())


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
    gender_path, age_path, age_gender_path = target_path+'/gender/', target_path+'/age/', target_path+'/age_gender/'
    print("target_path: ", target_path)

    # Which groups to augment:
    age_groups = {"0-2":0, "4-6":0, "8-13":0, "15-20":0}

    # How many samples to move from adience (for each age_group):
    NUM_SAMPLES_TRAIN = 180
    NUM_SAMPLES_VALID = 20

    # Load all the adience image info from ./datasets/raw/*.txt files into a single numpy matrix:
    adience_images, adience_img_count, col_names = load_all_folds_info(raw_path, age_groups)

    # For each age_group, sample images from adience, and move to wiki:
    counts = {}
    already_sampled = {}
    num_deleted = 0
    for age_group, age_group_count in age_groups.items():
        print("Sampling from adience for age_group: ", age_group)
        samples_train = sample_adience_images(adience_images, col_names, NUM_SAMPLES_TRAIN, age_group, already_sampled)
        num_deleted += augment_and_move(
            samples_train
            , datagen
            , counts
            , gender_path, age_path, age_gender_path
            , "train"
        )
        samples_valid = sample_adience_images(adience_images, col_names, NUM_SAMPLES_VALID, age_group, already_sampled)
        num_deleted += augment_and_move(
            samples_valid
            , datagen
            , counts
            , gender_path, age_path, age_gender_path
            , "valid"
        )

        print("Finished augment&move for age_group:'{}'. Deleted from adience: {}".format(age_group, num_deleted))
        print("Num train images sampled: ", sum([len(v[2]) for k,v in samples_train.items()]))
        print("Num train images sampled: ", sum([len(v[2]) for k,v in samples_valid.items()]))
        print()

    print("augmented counts: ", counts)
    print("Total Deleted from adience: ", num_deleted)
    print("Done! ")


if __name__ == '__main__':
    print("Test mode enabled? ", TEST_MODE)
    main()
    # explore_data()
