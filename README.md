# CIGA
CS194 Project - Team CIGA


To get up and running you'll first need to setup the data. This includes downloading three datasets (~12GB total), and then running some script to process, augment, and partition the data so that it's ready for use by our Keras code. See ./setup.md for instructions on how to do this. 

Once you have the data set up, you can reproduce our results. The Jupyter notebooks were purely for our exploration so you most likely won't find much of interest in those. Here is a guide to our code, assuming you've cloned our repo into a folder named ./ciga/: 

1. ./ciga/environment.yml, and ./ciga/environment-gpu.yml
	1. Our conda environments, for CPU and GPU machines respectively (details on setting these up are in ./ciga/setup.md)
2. ./ciga/datasets/raw/ - our scripts download and extract the datasets here. Processed and augmented data goes into the ./ciga/datasets/processed/ subfolder. 
2. ./ciga/models/ 
	1. Contains all our model definitions. 
2. ./ciga/training/
	1. train\_AndreyNet.py (our initial name for "sameres" was AndreyNet, we should rename this), train\_resnet.py, and train_vgg.py are our hyperparameter search scripts. 
	2. train\_to\_convergence.py is the script we used to pre-train all our models against the IMDB+Wiki dataset. 
	3. Note: pre-trained weights along with lots of other training info gets saved for each trained model, under ./ciga/models/saved_models/
4. ./ciga/testing/ - contains k-fold cross validation script, and some helper scripts that have functions for things like generating confusion matrices. 