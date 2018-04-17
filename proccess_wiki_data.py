import os
import numpy as np
from scipy import io as sio
import pandas as pd
from shutil import copyfile
from datetime import datetime, timedelta

wiki_path = './datasets/raw/wiki_crop/wiki_crop/'
valid_ration = 10

def mat_date_convert(mat_date):
    return (timedelta(days=mat_date - 366) + datetime(1, 1, 1)).year

def copy_data(data, class_name, origin_path, dest_path, max_samples=None):
    copied = 0
    for _, row in data.iterrows():
        if max_samples and copied == max_samples:
            break
        path = row['full_path'][0]
        img_name = path.replace('/', '_')
        old_path = origin_path + path
        new_path = dest_path + class_name +'/' + img_name
        copyfile(old_path, new_path)
        copied += 1

def proccess_wiki_mat(path):
	raw = sio.loadmat(path)
	raw = raw['wiki']
	raw = raw[0][0]

	data = []
	for i in raw:
	    data.append(i)
	data = np.array(data)
	data = data.squeeze()

	data = data.T
	data.shape

	df = pd.DataFrame(data)
	df.columns = ['dob', 'photo_taken', 'full_path', 'gender', 'name', 'face_location', 'face_score', 'second_face_score']

	df = df[df['face_score'] != float('-inf')]
	df = df.dropna(subset=['gender'])
	df = df[pd.isnull(df['second_face_score'])]

	df['age'] = df['photo_taken'] - df['dob'].apply(mat_date_convert)
	return df[['full_path', 'gender', 'age']]

def make_wiki_gender_dataset(data):
	train_directory = './datasets/processed/wiki/gender/train/'
	valid_directory = './datasets/processed/wiki/gender/valid/'

	# creates directory ifs not there already
	# Train:
	if not os.path.exists(train_directory + 'male'):
	    os.makedirs(train_directory + 'male')
	if not os.path.exists(train_directory + 'female'):
	    os.makedirs(train_directory + 'female')
	# Validate:
	if not os.path.exists(valid_directory + 'male'):
	    os.makedirs(valid_directory + 'male')
	if not os.path.exists(valid_directory + 'female'):
	    os.makedirs(valid_directory + 'female')

	num_male = int(sum(data['gender']))
	num_female = len(data) - num_male
	num_samples = min(num_female, num_male)

	female_samples = data[data['gender'] == 0]
	male_samples = data[data['gender'] == 1]

	train_female, val_female = female_samples[num_samples//valid_ration:num_samples], female_samples[:num_samples//valid_ration]
	train_male, val_male = male_samples[num_samples//valid_ration:num_samples], male_samples[:num_samples//valid_ration]

	copy_data(train_female, 'female', wiki_path, train_directory, num_samples)
	copy_data(val_female, 'female', wiki_path, valid_directory, num_samples)
	copy_data(train_male, 'male', wiki_path, train_directory, num_samples)
	copy_data(val_male, 'male', wiki_path, valid_directory, num_samples)

	print('male training:', len(train_male), 'validation:', len(val_male))
	print('female training:', len(train_female), 'validation:', len(val_female))

def make_wiki_age_dataset(data, age_partitions):
	train_directory = './datasets/processed/wiki/age/train/'
	valid_directory = './datasets/processed/wiki/age/valid/'

	for age_partition in age_partitions:
		lower, upper = age_partition
		class_name = str(lower) + '-' + str(upper)

		# Train:
		if not os.path.exists(train_directory + class_name):
		    os.makedirs(train_directory + class_name)
		# Validate:
		if not os.path.exists(valid_directory + class_name):
		    os.makedirs(valid_directory + class_name)

		mask = (data['age'] > lower) & (data['age'] < upper)
		masked = data[mask]
		num_samples = masked.shape[0]
		thresh = num_samples // valid_ration
		train, validate = masked[thresh:], masked[:thresh]
		copy_data(train, class_name, wiki_path, train_directory)
		copy_data(validate, class_name, wiki_path, valid_directory)

		print(class_name, 'training:', len(train), 'validation:', len(validate))

def make_wiki_age_gender_dataset(data, age_partitions):
	train_directory = './datasets/processed/wiki/age_gender/train/'
	valid_directory = './datasets/processed/wiki/age_gender/valid/'

	genders = ['female', 'male']

	for i in range(len(genders)):
		gender = genders[i]
		by_gender = data[data['gender'] == i]
		for age_partition in age_partitions:
			lower, upper = age_partition
			class_name = gender + '_' +str(lower) + '-' + str(upper)

			# Train:
			if not os.path.exists(train_directory + class_name):
			    os.makedirs(train_directory + class_name)
			# Validate:
			if not os.path.exists(valid_directory + class_name):
			    os.makedirs(valid_directory + class_name)

			age_mask = (by_gender['age'] > lower) & (by_gender['age'] < upper)
			masked = by_gender[age_mask]

			num_samples = masked.shape[0]
			thresh = num_samples // valid_ration
			train, validate = masked[thresh:], masked[:thresh]
			copy_data(train, class_name, wiki_path, train_directory)
			copy_data(validate, class_name, wiki_path, valid_directory)

			print(class_name, 'training:', len(train), 'validation:', len(validate))


def create_wiki_datasets(age=True, gender=True, gender_age=True):
	proccesed_data = proccess_wiki_mat('./datasets/raw/wiki/wiki/wiki.mat')
	age_partitions = [[0, 2], [4, 6], [8, 13], [15, 20], [25, 32], [38, 43], [48, 53], [60, 130]]
	if age:
		make_wiki_gender_dataset(proccesed_data)
		print("Done creating gender dataset")
		print('=================================================================')
	if gender:
		make_wiki_age_dataset(proccesed_data, age_partitions)
		print("Done creating age dataset")
		print('=================================================================')
	if gender_age:
		make_wiki_age_gender_dataset(proccesed_data, age_partitions)
		print("Done creating age+gender dataset")
		print('=================================================================')

if __name__ == '__main__':
	create_wiki_datasets(age=True, gender=True, gender_age=True)