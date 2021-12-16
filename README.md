### CS-503 Visual Intelligence

# One Shot Deep Motion Correction

---

This repository provides a framework for correcting human motion, showcased on the example of yoga poses.

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
[jupyter notebook](TODO_insert_showcase_jupyter_notebook).

If you are interested in rerunning the individual steps we display there, this is how you would do it.

### The image acquisition

### The pose estimation

### The pose classification into poses


### The good - bad pose classification

### The pose correction

#### The Learned Angle Distribution correction

#### The Nearest Neighbour correction


#### The Generative GAN correction


## Run Parameters Explained
TODO merge with run above
For running a test you specify all the parameters in lists. We then take all the permutations of these parameters and run these sequentially.

```[Potential options listed in a list]```

**graph_list:** ```["FullyConnectedGraph", "BinomialGraph", "RingOfCliques", "CirculantGraph", "CycleGraph", "Torus2D"]``` Choice of network topology. This decides how the nodes are laid out and who communicates with who. <br/>
**task_list:** ```["MNIST"]``` Which dataset to use. Currently only implemented MNIST <br/>
**nr_node_list:**  ```[Natural Numbers]``` Number of nodes to create. Note certain network topologies can only have a certain number of nodes. <br/>
**nr_classes_list:** ```[0-10] for MNIST``` Only has an effect if data_distribution is "non_iid". Determines the number of class labels that are given to each node. <br/>
**data_distribution:** ```['uniform', 'random', 'non_iid_uniform', 'non_iid_random']``` How data is split across nodes. Uniform means that nodes are given the same number of samples, random means the samples are randomly partitioned. non_iid means we do not shuffle the data and give each node only a certain number of class labels as defined by **nr_classes_list**.<br/>
**lr_list:** ```[Real Numbers]``` learning rate<br/>
**training_epochs:** ```[Natural Numbers]``` number of training epochs to run<br/>
**test_granularity:** ```[Natural Numbers]``` frequency with which to test the network on the test data set. Corresponds to test_granularity % epochs<br/>
**add_privacy_list:** ```[Boolean]``` Boolean flag to add differential privacy<br/>
**epsilon_list:** ```[Real Numbers]``` Only has effect if add_privacy_list is True. Quantifies the privacy properties of the DP-SGD algorithm. More [info](https://opacus.ai/docs/faq). <br/>
**delta_list:** ```[Real Numbers]``` Only has effect if add_privacy_list is True. Quantifies the privacy properties of the DP-SGD algorithm. More [info](https://opacus.ai/docs/faq). <br/>
**subset:** ```[Boolean]``` Whether or not to train on 30% of the training data. Used to save time. <br/>



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
|   |-- test_cGAN.py
|   |-- test_cLimbGAN.py
|-- notebooks -- |
|   |-- __init__.py
|   |-- pose_correction_pipeline.ipynb
|-- pose -- |
|   |-- __init__.py
|   |-- nearest_neighbor_correction.py
|   |-- plot.py
|   |-- pose_mediapipe.py
|   |-- pose_utils.py
|-- saved_model -- |
|   |-- cLimbGAN -- |
|   |   |-- model_after_epoch_780.pth
|   |-- mlp -- |
|   |   |-- 2021_11_22_20_46_21.ckpt
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
* `nearest_neighbor_correction.py`: functions needed for the nearest neighbour correction
* `plot.py`: functions needed to plot the image data and coordinate data
* `pose_mediapipe.py`: extract the poses from raw image data with mediapipe
* `pose_utils.py`: utilities needed for the pose extraction


## Authors
[Alexander Glavackij](https://github.com/alxglvckij),
[Alec Flowers](https://github.com/alec-flowers), 
[Nina Mainusch](https://github.com/Nina-Mainusch)
