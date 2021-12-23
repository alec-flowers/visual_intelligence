### CS-503 Visual Intelligence

# 3D Pose Correction

---

We provide a framework for correcting 3D human poses, showcased on the example of yoga poses.

[Accompanying Report](TODO INSERT LINK)

## Software dependencies

The code contained in this repository was tested on the following configuration of python:

`torch>=1.10.0`
`numpy>=1.21.3`
`matplotlib>=3.4.3`
`Pillow>=8.4.0`
`beautifulsoup4>=4.10.0`
`requests>=2.26.0`
`selenium>=4.0.0`
`pandas>=1.3.4`
`torchvision>=0.11.1`
`seaborn>=0.11.2`
`sklearn>=0.0`
`scikit-learn>=1.0.1`
`opencv-python>=4.5.4.58`
`mediapipe>=0.8.8.1`
`tqdm>=4.62.3`
`scipy>=1.7.2`
`notebook>=6.4.5`
`pytorch-lightning>=1.5.4`

## Installation Guide
#### Packages
```bash
pip3 install -r requirements.txt
```

#### Data Download
1. To use our data go to [DATA LINK](https://drive.google.com/drive/folders/1JxBM7r1Y8j3aFGCrnUgBDihJm51d70NA?usp=sharing)
and download the `train_test_pickled_data.zip`. Note it is 2.2 GB. 

2. Place this into the `data` folder and unzip.

3. Move the folders `train`, `test`, and `pickled_data` out of the unzip folder and into the data folder. These are 
necessary to run the code and if you don't have these folders `pose_utils.py` will complain. 

## Running our Code

You can run a showcase of our correction system in this 
[jupyter notebook](https://gitlab.com/aglavac/cs-503-visual-intelligence/-/blob/main/notebooks/pose_correction_pipeline.ipynb).

You can add your own images under the test folder to see how it works on you!
1. Take a photo of yourself performing one of the 3 yoga poses.
2. Use .jpg or .jpeg format or you have to manually modify the filepath.
3. Add the photo to the correct folder under `test`. 0 represents an incorrect pose, 1 represents a correct pose.

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

### The pose type classification
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

### The pose quality classification
To classify the quality (good/ bad) of a given pose, we have again trained and tested an MLP.
To train the network yourself, or to resume training a current model, or to just get the analyzing visualisations,
you can run the file ```classify_pose_quality.py``` in the ```classify``` folder.
It takes the following arguments:
```bash
-pickles "Path to load the pickled dataframes from the pose estimation from." (str)
-scratch "Train the classifier from scratch or load a previously trained model."  (bool)
-version "If you don't train from scratch, specify the model version to be resumed for training." (str)
-save "Path to save and load the trained model to/ from." (str)
-epochs "How many epochs to train the model for." (int)
-viz "Visualize confusion matrices of the classifier and correctly and incorrectly classified images." (bool)
```
There are defaults set for every argument, and we provide you with a pretrained model already, 
in case you just want to replicate our paper's figures.

### The pose correction
We have developed three approaches to correct a misclassified image. 
Here we describe how to run them with our code base.
#### The Learned Angle Distribution correction

We have implemented this correction in the `pose_correction_pipeline.ipynb`. The function `plot_distribution_with_image()`
plots the distribution with confidence intervals as red lines and the input image angles as blue lines.

The correct dataset which we build the distribution is loaded in from a pickle `pose_landmark_all_df`. This is created
by running the Pose Estimation piece from above. 

#### The Nearest Neighbour correction
We have implemented this correction in the `pose_correction_pipeline.ipynb`. The function `compare_two_figures()`
displays both the input image and the nearest neighbor image along with the 3D correction. 

The nearest neighbor dataset comes from `pose_landmark_numpy.pickle`. This is created by running the Pose Estimation
piece from above. 

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
|   |-- classify_pose_quality.py
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
|   |-- pose quality mlp -- |
|   |   |-- 2021_12_22_08_58.ckpt
|-- .gitignore
|-- README.md

```

## Files
**Classifier**
* `classifier_models.py`: the neural network classifier architectures
* `classify.py`: train the pose type MLP classifier from scratch or load a trained version
* `classify.py`: train the pose quality MLP classifier from scratch or load a trained version
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
* `plots_for_report`: various plots used in the report
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
