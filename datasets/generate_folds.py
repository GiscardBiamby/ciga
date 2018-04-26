import numpy as np, os, shutil, glob, sys, argparse
from random import shuffle


def folderize(dataset, k, label_type):
    root = os.path.dirname(os.path.abspath(__file__))
    outside_folds = os.path.join(root, "processed/" + dataset + "/" + label_type)
    train_paths, valid_paths, fold_paths = outside_folds, None, outside_folds + "/fold_"
    fold_paths = [fold_paths + str(i) for i in range(1, k+1)]
    to_train_paths, to_valid_paths = [fold + '/train' for fold in fold_paths], [fold + '/valid' for fold in fold_paths]
    fold_paths = to_train_paths + to_valid_paths

    if dataset in ["imdb", "wiki"]:
        train_paths += "/train"
        valid_paths = outside_folds + "/valid"
    elif dataset != "adience":
        print("Please enter either 'adience', 'imdb' or 'wiki' as the first argument.")
        return

    if label_type == "age":
        labels = ['/0-2', '/4-6', '/8-12', '/15-20', '/25-32', '/38-43', '/48-53', '/60-130']
    elif label_type == "gender":
        labels = ['/female', '/male']
    elif label_type == "age_gender":
        labels = ['/female_0-2', '/female_4-6', '/female_8-12', '/female_15-20',
                  '/female_25-32', '/female_38-43', '/female_48-53', '/female_60-130']
        labels += ['/male_0-2', '/male_4-6', '/male_8-12', '/male_15-20',
                   '/male_25-32', '/male_38-43', '/male_48-53', '/male_60-130']
    else:
        print("Please enter either 'age', 'gender' or 'age_gender' as the third argument.")
        return

    temp, fold_paths = fold_paths[:], []
    for fold in temp:
        fold_paths += [fold + label for label in labels]

    outside_folds = [train_paths + label for label in labels]
    if valid_paths:
        outside_folds += [valid_paths + label for label in labels]

    from_paths = []
    for path in outside_folds:
        from_paths += list(glob.iglob(os.path.join(path, "*")))
    shuffle(from_paths)

    for fold in fold_paths:
        if not os.path.exists(fold):
            os.makedirs(fold)

    n = len(from_paths)
    folds = [(from_paths[:int(i*n/k)]+from_paths[int((i+1)*n/k):], from_paths[int(i*n/k):int((i+1)*n/k)]) for i in range(k)]

    for i in range(k):
        train, valid = folds[i]
        to_train, to_valid = to_train_paths[i], to_valid_paths[i]
        for from_path in train:
            lst = from_path.split('/')
            to_path = to_train + '/' + lst[-2] + '/' + lst[-1]
            shutil.copy(from_path, to_path)
        for from_path in valid:
            lst = from_path.split('/')
            to_path = to_valid + '/' + lst[-2] + '/' + lst[-1]
            shutil.copy(from_path, to_path)

def main():
    parser = argparse.ArgumentParser(description='Break up processed datasets into folds.')
    parser.add_argument('dataset', help='Dataset to break into folds, either "adience", "imdb" or "wiki"')
    parser.add_argument('k', help='The number of folds to use, typically k=5 or k=7')
    parser.add_argument('label_type', help="The label type to be used, either: 'age', 'age_gender' or 'gender'")
    args = parser.parse_args()
    folderize(args.dataset, int(args.k), args.label_type)

if __name__ == '__main__':
    main()

