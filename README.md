### CS-503 Visual Intelligence

# 3D Pose Correction

---

We provide a framework for correcting 3D human poses, showcased on the example of yoga poses.

[Accompanying Report](TODO INSERT LINK)

## Software dependencies

The code contained in this repository was tested on the following configuration of python:

TODO insert dependencies

## Installation Guide

```bash
pip3 install -r requirements.txt
```

## Running our Code

You can run a showcase of our correction system in this 
[jupyter notebook](https://gitlab.com/aglavac/cs-503-visual-intelligence/-/blob/main/notebooks/pose_correction_pipeline.ipynb).

If you are interested in rerunning the individual steps we display there, this is how you would do it.

### The image acquisition
To extract images from google, you can run the ```scraping.py``` file. Before you can run it, you have
to 
1) Install Google Chrome (skip if its already installed)
2) Identify your Chrome version. Typically found by clicking About Google Chrome. 
 For example, if you currently have version 77.0.3865.90, your main version is 77, the number before the first dot.
3) Download the corresponding ChromeDriver from here for your main version and put the executable into an 
    accessible location (e.g. use Desktop/Scraping).
4) Install the Python Selenium package via ```pip install selenium```.

Then run the script with the following arguments:

```bash
-p "link to google image search" (str)
-k "web search keyword used to look up images"  (str)
-t "directory where the downloaded images will be stored" (str)
-n number of images to be scraped (int)
```
A successful scraping of downward dog images would for example be: 
```-p https://www.google.com/search?q=downward+dog&source=lnms&tbm=isch&sa=X&ved=2ahUKEwizuZvG8Oz0AhWJ7rsIHWgSDesQ_AUoAXoECAIQAw&biw=1244&bih=626&dpr=1.5 -k "downward dog" -t "./test" -n 15```

### The pose estimation
To estimate 3D poses on the acquired images with the library ```mediapipe``` 
you have to run the file ```pose_mediapipe.py``` in the ```pose``` folder.
Execute the file with the following (optional) arguments:
```bash
-path "Path to the images where you want to extract poses from." (str)
-save "Path to save the estimated poses and extracted dataframes to."  (str)
-viz "Visualize the data where you found poses and where you did not." (bool)
-skip "Skip saving the annotated images,  because they take a lot of memory." (bool)
```

We also provided default values such that the data gets pulled from a folder ```./data/train``` 
and saved to a folder ```./data/pickled_data``` and we automatically ```visualize``` and 
```don't skip``` the annotated images.

### The pose classification into poses
We have trained a multilayer perceptron (MLP) to classify our pose estimates into the poses 
downward dog, warrior 1 and warrior 2. To train it yourself, or to resume training a current model,
you can run the file ```classify.py``` in the ```classify``` folder.
It takes the following arguments:
```bash
-pickles "Path to load the pickled dataframes from the pose estimation from." (str)
-scratch "Train the classifier from scratch or load a previously trained model."  (bool)
-version "If you don't train from scratch, specify the model version to be resumed for training." (str)
-save "Path to save and load the trained model to/ from." (str)
-epochs "How many epochs to train the model for." (int)
-viz "Visualize confusion matrices of the classifier and correctly and incorrectly classified images." (bool)
```
There are defaults set for every argument, so it aligns with the rest of our pipeline.

### The good - bad pose classification
TODO

### The pose correction
We have developed three approaches to correct a misclassified image. 
Here we describe how to run them with our code base.
#### The Learned Angle Distribution correction
TODO
#### The Nearest Neighbour correction
TODO

#### The Generative cLimbGAN correction
To train our cLimbGAN or resume training of your current version of the model, 
run the ```cLimbGAN.py``` file from the ```gan``` folder.
We provide necessary default arguments in accordance with the prior pipeline, 
but if you want to chance something, here is how to:
```bash
-data "Path to the good poses that we want to train the cLimbGAN on." (str)
-path "Path to load and save the model from."  (str)
-scratch "Train the GAN from scratch or resume training a previous model" (bool)
-version  "If you don't train from scratch, specify the model version to be resumed for training." (int)
-epochs "How many epochs to train the model for." (int)
```


## File Structure
Here is the file structure of the project:
```bash
Project
|
|-- classifier -- |
|   |-- __init__.py
|   |-- classifier_models.py
|   |-- classify.py
|   |-- cnn_classifier.py
|   |-- test_classifier.py
|   |-- train_classifier.py
|-- data -- |
|   |-- __init__.py
|   |-- data_loading.py
|   |-- data_labeling.py
|   |-- scraping.py
|   |-- yoga_82_image_downloader.py
|-- gan -- |
|   |-- __init__.py
|   |-- cGAN.py
|   |-- cLimbGAN.py
|   |-- gan_models.py
|   |-- results_cGAN.py
|   |-- results_cLimbGAN.py
|-- notebooks -- |
|   |-- __init__.py
|   |-- pose_correction_pipeline.ipynb
|-- pose -- |
|   |-- __init__.py
|   |-- decision_tree.py
|   |-- nearest_neighbor_correction.py
|   |-- plot.py
|   |-- pose_mediapipe.py
|   |-- pose_utils.py
|-- saved_model -- |
|   |-- cLimbGAN -- |
|   |   |-- model_after_epoch_780.pth
|   |-- mlp -- |
|   |   |-- 2021_12_22_10_05.ckpt
|   |   |-- mlp_intermediate.ckpt
|-- .gitignore
|-- README.md

```

## Files
**Classifier**
* `classifier_models.py`: the neural network classifier architectures
* `classify.py`: train the MLP classifier from scratch or load a trained version
* `cnn_classifier.py`: train the CNN classifier from scratch or load a trained version
* `test_classifier.py`: make predictions with the trained model
* `train_classifier.py`: train an MLP or CNN classifier 

**Data**
* `data_labeling.py`: use this file to label collected data
* `data_loading.py`: manage datasets and dataloaders
* `scraping.py`: scrape data from google with chrome
* `yoga_82_image_downloader.py`: script to download the images of the Yoga.82 data set

**GAN**
* `cGAN.py`: GAN conditioned on the pose labels
* `cLimbGAN.py`: GAN conditioned on the pose labels and limb lengths
* `gan_models.py`: the generator and discriminator architectures
* `test_cGAN.py`: generate poses with the cGAN
* `test_cLimbGAN.py`: generate poses with the cLimbGAN

**Notebooks**
* `pose_correction_pipeline.ipynb`: a showcase of our whole correction system

**Pose**
* `decision_tree.py`: functions needed for creating a decision tree for good/ bad pose execution
* `nearest_neighbor_correction.py`: functions needed for the nearest neighbour correction
* `plot.py`: functions needed to plot the image data and coordinate data
* `pose_mediapipe.py`: extract the poses from raw image data with mediapipe
* `pose_utils.py`: utilities needed for the pose extraction


## Authors
[Alexander Glavackij](https://github.com/alxglvckij),
[Alec Flowers](https://github.com/alec-flowers), 
[Nina Mainusch](https://github.com/Nina-Mainusch)
