
import os, random, shutil, glob
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np

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
        fold = np.genfromtxt(os.path.join(raw_path, "fold_%d_data.txt" % j), usecols=(0, 1, 2, 3, 4),
                             delimiter='\t', dtype=None)

        base_dir_column = np.array([os.path.join(raw_path, "faces/faces/")] * fold.shape[0]).reshape((fold.shape[0], 1))
        fold = np.hstack((fold, base_dir_column))
        folds.append(fold)

        fold = np.genfromtxt(os.path.join(raw_path, "fold_frontal_%d_data.txt" % j), usecols=(0, 1, 2, 3, 4),
                             delimiter='\t',
                             dtype=None)
        base_dir_column = np.array([os.path.join(raw_path, "aligned/aligned/")] * fold.shape[0]).reshape(
            (fold.shape[0], 1))
        fold = np.hstack((fold, base_dir_column))
        folds.append(fold)
    adience_images = np.vstack(folds)
    # print("adience_images.shape: ", adience_images.shape)

    # Filter some data:
    before = adience_images.shape[0]
    keep = [False]*adience_images.shape[0]
    for idx, img in enumerate(adience_images):
        age = get_age(img)
        if age is not None and age in age_groups.keys():
            keep[idx] = True
    adience_images = adience_images[keep]
    after = adience_images.shape[0]
    # print("Removed {} images on age filtering".format(after-before))


    adience_img_count = adience_images.shape[0]
    return adience_images, adience_img_count


def sample_adience_images(adience_images, num_samples, age_group):
    """

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

    # Choose num_samples images to move:
    adience_agegroup_indexes = list(range(adience_agegroup.shape[0]))
    sampled_indexes_candidates = np.random.choice(adience_agegroup_indexes, size=num_samples, replace=False)

    # Gather paths of images we will augment and move:
    sampled_images = []  # images we will augment and move
    sampled_indexes = []
    sampled_count = 0

    for idx in sampled_indexes_candidates:
        img = adience_images[idx]
        if sampled_count >= num_samples:
            break
        img_path = os.path.join(img[-1], img[0])

        # Each image from the adience_images might have multiple coarse_tilt_aligned versions on disk,
        # so this glob might return multiple files for one image from the images numpy array (hence the break
        # condition at top of loop):
        sampled_images.append({"paths": glob.glob("{}/*{}".format(img_path, img[1])), "img": img })
        sampled_count += len(sampled_images[-1]["paths"])
        sampled_indexes.append(idx)

    print("age_group:'{}', #sampled_imgs: {}, #sampled_idxs: {}".format(age_group, sampled_count, len(sampled_indexes)))
    return sampled_images


def get_gender(img):
    gender = img[4]
    return gender if len(gender) > 0 else None


def get_age(img):
    age = img[3]
    if age is not None and len(age) > 5:
        age = age[1:-1].replace(', ', '-')
        return age
    return None


def augment_img(datagen, target_dir, img, counts):
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

    if target_dir not in counts:
        counts[target_dir] = 0

    for augmented in datagen.flow(img, batch_size=1, save_to_dir=target_dir, save_prefix='aug', save_format='jpeg'):
        if aug_count >= NUM_AUG:
            break
        aug_count += 1
        counts[target_dir] += 1
    # print("\taug_count: ", counts[target_dir])


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

    num_deleted = 0
    processed = []
    for sample in sampled_images:
        img_paths, img_data = sample["paths"], sample["img"]
        for img_path in img_paths:
            print("augmenting img: ", img_path)
            img = load_img(img_path)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            gender, age = get_gender(img_data), get_age(img_data)
            if gender=="m":
                augment_img(datagen, male_dir, x, counts)
                augment_img(datagen, age_dir + age, x, counts)
                augment_img(datagen, age_gender_dir + 'male_' + age, x, counts)
            elif gender=="f":
                augment_img(datagen, female_dir, x, counts)
                augment_img(datagen, age_dir + age, x, counts)
                augment_img(datagen, age_gender_dir + 'female_' + age, x, counts)
            else:
                augment_img(datagen, age_dir + age, x, counts)
            processed.append(img_path)
            num_deleted += 1

    print("Deleting {} imgs from adience raw".format(len(processed)))
    for img in processed:
        if os.exists(img):
            os.remove(img)

    return num_deleted




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
    processed_path = os.path.join(root, "processed/wiki")

    gender, age, age_gender = processed_path + '/gender', processed_path + '/age', processed_path + '/age_gender'
    paths = [gender, age, age_gender] + [gender + '/female', gender + '/male']
    paths += [age + p for p in ['/0-2', '/4-6', '/8-12', '/15-20', '/25-32', '/38-43', '/48-53', '/60-100']]
    paths += [age_gender + p for p in ['/male_0-2', '/male_4-6', '/male_8-12', '/male_15-20', '/male_25-32', '/male_38-43', '/male_48-53', '/male_60-100']]
    paths += [age_gender + p for p in ['/female_0-2', '/female_4-6', '/female_8-12', '/female_15-20', '/female_25-32', '/female_38-43', '/female_48-53', '/female_60-100']]

    for p in paths:
        print("path: ", p)
        if not os.path.exists(p):
            os.makedirs(p)

    male, female, age_path = processed_path + '/gender/male', processed_path + '/gender/female', processed_path + '/age/'
    print("processed_path: ", processed_path)
    print("male, female, age_path : ", male, female, age_path )

    age_groups = {"0-2":0, "4-6":0, "8-12":0, "15-20":0}
    # how many samples to move from adience, for each age_group:
    NUM_SAMPLES = 200


    # Load all the adience image info into a single numpy matrix:
    adience_images, adience_img_count = load_all_folds_info(raw_path, age_groups)


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

main()

# wiki before: 19449
# wiki after: 20131