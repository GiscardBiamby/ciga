## 1. Create Environment 

Our project uses python virtual env named "ciga". To create the environment and install the dependencies, run: 

```bash
conda env create -f environment.yml
```

If environment.yml gets updated, you'll need to update your environment. In which case you can run: 

```bash
conda env update -f environment.yml
```

#### 1b. (Optional) Install direnv

https://direnv.net/

This tool automatically activates the "ciga" virtual environment when you navigate to your project folder, and deactivates it if you navigate away from the folder, saving you the trouble of having to call "source activate ciga" and "source deactivate". If you want to use it, install it using instructions from the website.

You may have to run `direnv allow .` the first time (or whenever .envrc changes, which shouldn't be often if ever). 

## 2. Download datasets
Run the following command (only tested w/ Python 3.6, e.g., "ciga" virtual env): 

```bash
cd datasets
python download_data.py
```

This will download and extract the datasets if you don't already have them. 

**Note: ** By default this currently only downloads the wiki datsets. You have to modify flags in the `getDatasets()`function call at the bottom of download_data.py to enable downloading the IMDB ones. It is all set up to download the IMDB data but I simply made it default to not download those because they are very large and I haven't needed to use that data (yet?). 

The files are extracted into `./<project-root>/datasets/<filename>/*`, where ``<filename>` is the name of the corresponding downloaded archive (minus file extension). 


## 3. Process Datasets

### 3a. Partition Data:

The keras code uses .flow_from_directory(), which expects all images to be in subfolders according to their ground-truth labels. These scripts partition the various datasets according to that expectation. The partitioned data goes into ./datasets/processed/[dataset\_name]/[partition\_scheme]/. [partition\_scheme] describes a certain partitioning scheme for the data. There are three partition schemes: by age, by gender, or by age_gender (both together).

```bash
cd datasets
python populate_subfolders.py
python process_wiki_imdb_data.py [--createMerged]
```
Note: the --createMerged flag is optional. By default the script doesn't create the combined imdb+wiki dataset, but if you specify this flag it will create imdb+wiki merged and put it in ./datasets/processed/imdbwiki/. 

### 3b. Augment data
Permanently move some data from adience dataset to the imdb/wiki ones so that the distributions are more even. 

```bash
cd datasets
python augment_data.py
```

### 3c. (Optional) K Folds Dataset Pre-processing

If you intend to use K folds cross validation on a specific dataset, please accomplish the following pre-processing step before attempting to do so:

```bash
cd datasets
python generate_folds.py dataset_name K label_name
```

Note: dataset_name should be either "imdb", "wiki", or "adience".
      label_name should be either "age", "gender", or "age_gender"
