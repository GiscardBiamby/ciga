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
python download_data.py
```

This will download and extract the datasets if you don't already have them. 

**Note: ** By default this currently only downloads the wiki datsets. You have to modify flags in the `getDatasets()`function call at the bottom of download_data.py to enable downloading the IMDB ones. It is all set up to download the IMDB data but I simply made it default to not download those because they are very large and I haven't needed to use that data (yet?). 

The files are extracted into `./<project-root>/datasets/<filename>/*`, where ``<filename>` is the name of the corresponding downloaded archive (minus file extension). 



