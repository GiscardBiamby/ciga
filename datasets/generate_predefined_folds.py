import numpy as np, os, shutil, glob, sys, argparse
from random import shuffle


def folderize(label_type):
    dataset = 'adience'
    root = os.path.dirname(os.path.abspath(__file__))
    outside_folds = os.path.join(root, "processed/" + dataset + "/" + label_type)
    train_paths, valid_paths, fold_paths = outside_folds, None, outside_folds + "_fold_"
    fold_paths = [fold_paths + str(i) for i in range(5)]

    # Grab info from fold text documents
    fold_txts = os.path.join(root, "raw/fold_")
    fold_txts = zip([fold_txts + "%d_data.txt" % i for i in range(5)], [fold_txts + "frontal_%d_data.txt" % i for i in range(5)])
    fold_txts = [np.vstack((np.genfromtxt(i, usecols=(0, 1, 2, 3, 4), delimiter='\t', dtype=None),
                            np.genfromtxt(j, usecols=(0, 1, 2, 3, 4), delimiter='\t', dtype=None))) for i, j in fold_txts]
    raw_path = os.path.join(root, "raw")

    if label_type == "age":
        values = ['/a_0-2', '/b_4-6', '/c_8-13', '/d_15-20', '/e_25-32', '/f_38-43', '/g_48-53', '/h_60-130']
        keys = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
    elif label_type == "gender":
        values = ['/female', '/male']
        keys = ['f', 'm']
    elif label_type == "age_gender":
        values = ['/a_female_0-2', '/b_female_4-6', '/c_female_8-13', '/d_female_15-20',
                  '/e_female_25-32', '/f_female_38-43', '/g_female_48-53', '/h_female_60-130']
        values += ['/i_male_0-2', '/j_male_4-6', '/k_male_8-13', '/l_male_15-20',
                   '/m_male_25-32', '/n_male_38-43', '/o_male_48-53', '/p_male_60-130']
        keys = ['f(0, 2)', 'f(4, 6)', 'f(8, 12)', 'f(15, 20)',
                  'f(25, 32)', 'f(38, 43)', 'f(48, 53)', 'f(60, 100)']
        keys += ['m(0, 2)', 'm(4, 6)', 'm(8, 12)', 'm(15, 20)',
                   'm(25, 32)', 'm(38, 43)', 'm(48, 53)', 'm(60, 100)']
    else:
        print("Please enter either 'age', 'gender' or 'age_gender' as the only argument.")
        return

    fold_idx = 0
    for fold in fold_txts:
        temp_values = [fold_paths[fold_idx] + '/valid' + val for val in values]
        for p in temp_values:
            if not os.path.exists(p):
                os.makedirs(p)
        val_dct = {k: v for k, v in zip(keys, temp_values)}
        temp_values = [[path + '/train' + val for path in fold_paths[:fold_idx] + fold_paths[fold_idx+1:]] for val in values]
        for t in temp_values:
            for p in t:
                if not os.path.exists(p):
                    os.makedirs(p)
        train_dct = {k: v for k, v in zip(keys, temp_values)}
        for i in range(1, fold.shape[0]):
            path, img, id = fold[i, 0].decode('UTF-8'), fold[i, 1].decode('UTF-8'), fold[i, 2].decode('UTF-8')
            face_path = raw_path + '/faces/faces/' + path
            aligned_path = raw_path + '/aligned/aligned/' + path
            from_paths, to_paths = list(glob.iglob(os.path.join(face_path, "*" + id + "." + img))), []
            from_paths += list(glob.iglob(os.path.join(aligned_path, "*" + id + "." + img)))
            if label_type == 'gender':
                gender = fold[i, 4].decode('UTF-8')
                if gender is None or len(gender) < 1:
                    continue
                to_paths = [val_dct[gender]] + train_dct[gender]
            elif label_type == 'age':
                age = fold[i, 3].decode('UTF-8')
                if age is None or len(age) <= 5 or age not in val_dct.keys():
                    continue
                to_paths = [val_dct[age]] + train_dct[age]
            elif label_type == 'age_gender':
                age = fold[i, 3].decode('UTF-8')
                if age is None or len(age) <= 5 or age not in val_dct.keys():
                    continue
                gender = fold[i, 4].decode('UTF-8')
                if gender is None or len(gender) < 1:
                    continue
                to_paths = [val_dct[gender+age]] + train_dct[gender+age]
            to_paths = [p + '/' + id + '.' + img for p in to_paths]
            for from_path in from_paths:
                for to_path in to_paths:
                    if 'course' in from_path:
                        to_path = to_path[:-3] + 'c' + '.jpg'
                    else:
                        to_path = to_path[:-3] + 'l' + '.jpg'
                    shutil.copy(from_path, to_path)
        fold_idx += 1


def main():
    parser = argparse.ArgumentParser(description='Break up processed datasets into folds.')
    parser.add_argument('label_type', help="The label type to be used, either: 'age', 'age_gender' or 'gender'")
    args = parser.parse_args()
    folderize(args.label_type)

if __name__ == '__main__':
    main()

