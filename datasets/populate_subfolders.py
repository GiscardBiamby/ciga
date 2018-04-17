import numpy as np, os, shutil, glob

root = os.path.dirname(os.path.abspath(__file__))
raw_path = os.path.join(root, "raw")
processed_path = os.path.join(root, "processed/adience")

gender, age, gender_age = processed_path + '/gender', processed_path + '/age', processed_path + '/gender_age'
paths = [gender, age, gender_age] + [gender + '/female', gender + '/male']
paths += [age + p for p in ['/0-2', '/4-6', '/8-13', '/15-20', '/25-32', '/38-43', '/48-53', '/60-100']]
paths += [gender_age + p for p in ['/male_0-2', '/male_4-6', '/male_8-13', '/male_15-20', '/male_25-32', '/male_38-43', '/male_48-53', '/male_60-100']]
paths += [gender_age + p for p in ['/female_0-2', '/female_4-6', '/female_8-13', '/female_15-20', '/female_25-32', '/female_38-43', '/female_48-53', '/female_60-100']]

for p in paths:
    if not os.path.exists(p):
        os.makedirs(p)

male, female, age_path = processed_path + '/gender/male', processed_path + '/gender/female', processed_path + '/age/'
for j in range(5):
    folds = [np.genfromtxt(os.path.join(raw_path, "fold_%d_data.txt" % j), usecols=(0,1,2,3,4), delimiter='\t', dtype=None)]
    folds += [np.genfromtxt(os.path.join(raw_path, "fold_frontal_%d_data.txt" % j), usecols=(0,1,2,3,4), delimiter='\t', dtype=None)]
    print("fold %d out of 5" % (j+1))
    for fold in folds:
        for i in range(1, fold.shape[0]):
            age = fold[i, 3].decode('UTF-8')
            if age is not None and len(age) > 5:
                age = age[1:-1].replace(', ', '-')
            else:
                age = None
            gender = fold[i, 4].decode('UTF-8')
            gender = gender if len(gender) > 0 else None
            path, img, id = fold[i, 0].decode('UTF-8'), fold[i, 1].decode('UTF-8'), fold[i, 2].decode('UTF-8')
            face_path = raw_path + '/faces/faces/' + path
            aligned_path = raw_path + '/aligned/aligned/' + path
            paths = list(glob.iglob(os.path.join(face_path, "*" + id + "." + img)))
            paths += list(glob.iglob(os.path.join(aligned_path, "*" + id + "." + img)))
            if gender == 'm':
                for src in paths:
                    shutil.copy(src, male)
                if age is not None:
                    for src in paths:
                        shutil.copy(src, age_path + age)
                        shutil.copy(src, gender_age + 'male_' + age)
            elif gender == 'f':
                for src in paths:
                    shutil.copy(src, female)
                if age is not None:
                    for src in paths:
                        shutil.copy(src, age_path + age)
                        shutil.copy(src, gender_age + 'female_' + age)
            elif age is not None:
                for src in paths:
                    shutil.copy(src, age_path + age)
