import urllib.request as ulib
import shutil
from pathlib import Path
import hashlib
import os.path
import sys, argparse, logging
import tarfile
import os, equests

def verifyChecksum(filepath, checksum):
    """
    Verifies the file checksum
    :param filepath: path to file
    :param checksum: expected checksum
    :return: True if file checksum matches the supplied checksum. False o/w.
    """
    if checksum is None:
        return True
    chunksize_bytes = 128*10
    hasher = hashlib.md5()

    print("verifying checksum for {}...".format(filepath))

    try:
        with open(filepath, 'rb') as file:
            for chunk in iter(lambda: file.read(chunksize_bytes), b""):
                hasher.update(chunk)
    except Exception as e:
        raise IOError("File '{}' exists but encountered an error while trying to calculate its checksum. Try deleting the file (it's probably partially downloaded).".format(filepath)) from e
    actual_checksum = hasher.hexdigest()
    print("Expected md5: ", checksum)
    print("Actual md5  : ", actual_checksum)
    return checksum == actual_checksum

def isFileDownloaded(filepath, checksum):
    """
    Checks if the file has already been downloaded, and is complete (valid checksum).
    :param filepath:
    :param checksum:
    :return:
    """
    file = Path(filepath)
    if not file.is_file():
        return False

    return verifyChecksum(filepath, checksum)

def downloadFile(url, filepath, checksum):
    print("Downloading {}...".format(url))
    if checksum is None:
        r = requests.get(url, auth=("adiencedb","adience"))
        if r.status_code == 200:
            with open(filepath, "wb") as out:
                for bits in r.iter_content():
                    out.write(bits)
    else:
        with ulib.urlopen(url) as response, open(filepath, "wb") as targetfile:
            shutil.copyfileobj(response, targetfile)
    return verifyChecksum(filepath, checksum)

def extractFile(filepath, filename, extract_path):
    # A text file for each archive is saved to indicate that it's already been extracted.
    extractedMarker = os.path.join(extract_path, filename + ".extracted.txt")
    if os.path.exists(extractedMarker):
        print("File has already been extracted. To force re-extraction, delete the file: '{}'".format(extractedMarker))
        return

    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    print("Extracting {} to path: {}...".format(filepath, extract_path))

    with tarfile.open(filepath) as tar:
       tar.extractall(path = extract_path)

    # Mark that the file has been extracted, so we don't try to do it again if this code gets run again:
    open(extractedMarker, 'a').close()

def createFolders(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

def getDataset(name, urls):
    """
    Downloads files associated with a dataset.
    :param urls: an array of tuples. Each tuple has 3 elements in the format (url, target_filename, expected_checksum)
    :return:
    """
    dataset_dir = os.path.join(os.getcwd(), "datasets")
    rawdata_dir = os.path.join(dataset_dir, "raw")
    processed_dir = os.path.join(dataset_dir, "processed")
    createFolders([dataset_dir, rawdata_dir, processed_dir])

    # Download:
    for url, filename, checksum in urls:
        filepath = os.path.join(rawdata_dir, filename)
        if not isFileDownloaded(filepath, checksum):
            downloadSuccess = downloadFile(url, filepath, checksum)
            if not downloadSuccess:
                raise IOError("Failed to download '{}'!".format(url))
        extract_dir = os.path.join(rawdata_dir, filename.replace(".tar.gz", "").replace(".tar", ""))
        extractFile(filepath, filename, extract_dir)

def getDatasets(wiki=True, imdbFull=False, imdbSmall=False, adience=False):

    ## Wiki images & metadata:
    wiki_dataset = [
        ("https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki.tar.gz", "wiki.tar.gz", "6d4b8474b832f13f7f07cfbd3ed79522")
        , ("https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar", "wiki_crop.tar", "f536eb7f5eae229ae8f286184364b42b")
    ]
    ## IMDB images:
    imdb_full_dataset = [
        ("https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_0.tar", "imdb_0.tar", "04fa5f97ea0b3a2c001875fd3403f493")
        , ("https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_1.tar", "imdb_1.tar", "38558992f7b2f7ee9bfebca8df0080c2")
        , ("https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_2.tar", "imdb_2.tar", "4552b93a2bf65dcb42e2b214fcd79d47")
        , ("https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_3.tar", "imdb_3.tar", "8f654d85827f3188a00369426dd70403")
        , ("https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_4.tar", "imdb_4.tar", "2c5fce7af03993c96f0bba8e28a72da2")
        , ("https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_5.tar", "imdb_5.tar", "c26753bd456f71677ba68753b5484c75")
        , ("https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_6.tar", "imdb_6.tar", "5ce6db6f3690670e403891671ab4aa04")
        , ("https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_7.tar", "imdb_7.tar", "843fd7c9a93516e5046d3c40967d2e4f")
        , ("https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_8.tar", "imdb_8.tar", "bbaad2d301a359bbad4da3ddd3dbed7f")
        , ("https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_9.tar", "imdb_9.tar", "8f6159fa59e2a2772f2c352e0e8dd8da")
    ]
    # IMDB concise
    # (not sure yet if we need both of these together, and also not sure if metadata has info for the full dataset,
    # the faces-only, or both).
    imdb_concise_dataset = [
        ## IMDB images metadata:
        ("https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_meta.tar", "imdb_meta.tar", "469433135f1e961c9f4c0304d0b5db1e")
        ## IMDB faces only:
        , ("https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar", "imdb_crop.tar", "44b7548f288c14397cb7a7bab35ebe14")
    ]

    ## Adience images + labels:
    adience_dataset = [
    ("http://www.cslab.openu.ac.il/personal/Hassner/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/(old%20a)%20fold_0_data.txt", "old_fold_0_data.txt", None),
    ("http://www.cslab.openu.ac.il/personal/Hassner/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/(old%20a)%20fold_1_data.txt", "old_fold_1_data.txt", None),
    ("http://www.cslab.openu.ac.il/personal/Hassner/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/(old%20a)%20fold_2_data.txt", "old_fold_2_data.txt", None),
    ("http://www.cslab.openu.ac.il/personal/Hassner/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/(old%20a)%20fold_3_data.txt", "old_fold_3_data.txt", None),
    ("http://www.cslab.openu.ac.il/personal/Hassner/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/(old%20a)%20fold_4_data.txt", "old_fold_4_data.txt", None),
    ("http://www.cslab.openu.ac.il/personal/Hassner/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/(old%20a)%20fold_frontal_0_data.txt", "old_fold_frontal_0_data.txt", None),
    ("http://www.cslab.openu.ac.il/personal/Hassner/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/(old%20a)%20fold_frontal_1_data.txt", "old_fold_frontal_1_data.txt", None),
    ("http://www.cslab.openu.ac.il/personal/Hassner/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/(old%20a)%20fold_frontal_2_data.txt", "old_fold_frontal_2_data.txt", None),
    ("http://www.cslab.openu.ac.il/personal/Hassner/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/(old%20a)%20fold_frontal_3_data.txt", "old_fold_frontal_3_data.txt", None),
    ("http://www.cslab.openu.ac.il/personal/Hassner/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/(old%20a)%20fold_frontal_4_data.txt", "old_fold_frontal_4_data.txt", None),
    ("http://www.cslab.openu.ac.il/personal/Hassner/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/LICENSE.txt", "LICENSE.txt", None),
    ("http://www.cslab.openu.ac.il/personal/Hassner/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/aligned.tar.gz", "aligned.tar.gz", None),
    ("http://www.cslab.openu.ac.il/personal/Hassner/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/faces.tar.gz", "faces.tar.gz", None),
    ("http://www.cslab.openu.ac.il/personal/Hassner/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/fold_0_data.txt", "fold_0_data.txt", None),
    ("http://www.cslab.openu.ac.il/personal/Hassner/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/fold_1_data.txt", "fold_1_data.txt", None),
    ("http://www.cslab.openu.ac.il/personal/Hassner/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/fold_2_data.txt", "fold_2_data.txt", None),
    ("http://www.cslab.openu.ac.il/personal/Hassner/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/fold_3_data.txt", "fold_3_data.txt", None),
    ("http://www.cslab.openu.ac.il/personal/Hassner/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/fold_4_data.txt", "fold_4_data.txt", None),
    ("http://www.cslab.openu.ac.il/personal/Hassner/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/fold_frontal_0_data.txt", "fold_frontal_0_data.txt", None),
    ("http://www.cslab.openu.ac.il/personal/Hassner/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/fold_frontal_1_data.txt", "fold_frontal_1_data.txt", None),
    ("http://www.cslab.openu.ac.il/personal/Hassner/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/fold_frontal_2_data.txt", "fold_frontal_2_data.txt", None),
    ("http://www.cslab.openu.ac.il/personal/Hassner/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/fold_frontal_3_data.txt", "fold_frontal_3_data.txt", None),
    ("http://www.cslab.openu.ac.il/personal/Hassner/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/fold_frontal_4_data.txt", "fold_frontal_4_data.txt", None)
    ]

    if wiki:
        getDataset("wiki", wiki_dataset)
    if imdbFull:
        getDataset("imdb", imdb_full_dataset)
    if imdbSmall:
        getDataset("imdb", imdb_concise_dataset)
    if adience:
        getDataset("adience", adience_dataset)

if __name__ == '__main__':
    getDatasets(wiki=True, imdbFull=False, imdbSmall=False, adience=False)
