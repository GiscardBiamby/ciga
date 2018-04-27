import os
import numpy as np
from scipy import io as sio
import pandas as pd
from shutil import copyfile
from datetime import datetime, timedelta
import argparse
import distutils.dir_util

valid_ration = 10

def mat_date_convert(mat_date):
    try:
        return (timedelta(days=mat_date - 366) + datetime(1, 1, 1)).year
    except Exception as e:
        return None

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

def proccess_mat(path, dict_key):
	raw = sio.loadmat(path)
	raw = raw[dict_key]
	raw = raw[0][0]

	data = []

	for idx, i in enumerate(raw):
		if not (dict_key == "imdb" and idx == 8):
			data.append(i)
	data = np.array(data)
	data = data.squeeze()

	data = data.T
	data.shape

	df = pd.DataFrame(data)
	if dict_key == "wiki":
		df.columns = ['dob', 'photo_taken', 'full_path', 'gender', 'name', 'face_location', 'face_score',
					  'second_face_score']
	else:
		df.columns = ['dob', 'photo_taken', 'full_path', 'gender', 'name', 'face_location', 'face_score',
					  'second_face_score', 'celeb_names']

	totalRows = df.shape[0]
	print("{} total rows: {}".format(dict_key, totalRows))

	before = df.shape[0]
	df = df[df['face_score'] != float('-inf')]
	print("...Dropped {} rows with null or not-convertible face_score.".format(before - df.shape[0]))

	before = df.shape[0]
	df = df.dropna(subset=['gender'])
	print("...Dropped {} rows with null gender.".format(before - df.shape[0]))

	before = df.shape[0]
	df = df[pd.isnull(df['second_face_score'])]
	print("...Dropped {} rows with more than one face.".format(before - df.shape[0]))

	before = df.shape[0]
	df["dob"] = df['dob'].apply(mat_date_convert)
	df = df[~pd.isnull(df['dob'])]
	print("...Dropped {} rows with null or not-convertible DOB...".format(before - df.shape[0]))

	totalRowsAfter = df.shape[0]
	print("Finished scrubbing {}. totalRowsAfter: {}, total dropped: {}".format(
		dict_key, totalRowsAfter, (totalRows - totalRowsAfter))
	)

	df['age'] = df['photo_taken'] - df['dob']
	return df[['full_path', 'gender', 'age']]

def make_gender_dataset(raw_images_path, dataset_name, data):
	train_directory = './processed/{}/gender/train/'.format(dataset_name)
	valid_directory = './processed/{}/gender/valid/'.format(dataset_name)
	print("train dir: {}, valid_dir: ".format(train_directory, valid_directory))
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

	copy_data(train_female, 'female', raw_images_path, train_directory, num_samples)
	copy_data(val_female, 'female', raw_images_path, valid_directory, num_samples)
	copy_data(train_male, 'male', raw_images_path, train_directory, num_samples)
	copy_data(val_male, 'male', raw_images_path, valid_directory, num_samples)

	print('male training:', len(train_male), 'validation:', len(val_male))
	print('female training:', len(train_female), 'validation:', len(val_female))

def make_age_dataset(raw_images_path, dataset_name, data, age_partitions):
	train_directory = './processed/{}/age/train/'.format(dataset_name)
	valid_directory = './processed/{}/age/valid/'.format(dataset_name)
	print("train dir: {}, valid_dir: ".format(train_directory, valid_directory))

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
		copy_data(train, class_name, raw_images_path, train_directory)
		copy_data(validate, class_name, raw_images_path, valid_directory)

		print(class_name, 'training:', len(train), 'validation:', len(validate))

def make_age_gender_dataset(raw_images_path, dataset_name, data, age_partitions):
	train_directory = './processed/{}/age_gender/train/'.format(dataset_name)
	valid_directory = './processed/{}/age_gender/valid/'.format(dataset_name)

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
			copy_data(train, class_name, raw_images_path, train_directory)
			copy_data(validate, class_name, raw_images_path, valid_directory)

			print(class_name, 'training:', len(train), 'validation:', len(validate))


def create_datasets(metadata_path, raw_images_path, metadata_dict_key, age=True, gender=True, gender_age=True):
	proccesed_data = proccess_mat(metadata_path, metadata_dict_key)
	age_partitions = [[0, 2], [4, 6], [8, 12], [15, 20], [25, 32], [38, 43], [48, 53], [60, 100]]
	if age:
		make_gender_dataset(raw_images_path, metadata_dict_key, proccesed_data)
		print("Done creating gender dataset")
		print('=================================================================')
	if gender:
		make_age_dataset(raw_images_path, metadata_dict_key, proccesed_data, age_partitions)
		print("Done creating age dataset")
		print('=================================================================')
	if gender_age:
		make_age_gender_dataset(raw_images_path, metadata_dict_key, proccesed_data, age_partitions)
		print("Done creating age+gender dataset")
		print('=================================================================')

def create_merged():
	print('=================================================================')
	print("Creating imdb+wiki dataset...")
	imdb_dir = "./processed/imdb/"
	wiki_dir = "./processed/wiki/"
	target_dir = "./processed/imdbwiki/"

	if not os.path.exists(target_dir):
		os.makedirs(target_dir)

	print("Copying imdb...")
	distutils.dir_util.copy_tree(imdb_dir, target_dir)
	print("Copying wiki...")
	distutils.dir_util.copy_tree(wiki_dir, target_dir)


	print('=================================================================')

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Setup folder structure for imdb and wiki face datasets for use by '
												 'keras flow_from_directory(), so that images are organized on disk '
												 'by subfolders inside of subfolders like: '
												 './datasets/processed/<dataset_name>/<age_or_gender'
												 '>/<ground_truth_label>/.')
	parser.add_argument(
		'--createMerged'
		, default=False
		, action='store_true'
		, required=False
		, help='If specified, will also combine the two datasets into one "imdbwiki" dataset. Default is False '
			   'because it can take up a lot of space.'
	)
	args = parser.parse_args()
	create_datasets(
        metadata_path='./raw/wiki/wiki/wiki.mat'
        , raw_images_path = './raw/wiki_crop/wiki_crop/'
        , metadata_dict_key="wiki"
        , age=True
        , gender=True
        , gender_age=True
	)
	create_datasets(
        metadata_path='./raw/imdb_meta/imdb/imdb.mat'
        , raw_images_path = './raw/imdb_crop/imdb_crop/'
        , metadata_dict_key="imdb"
        , age=True
        , gender=True
        , gender_age=True
	)
	if args.createMerged:
		create_merged()