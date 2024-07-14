# Solar Panel Fault Detection - Proof of Concept (POC)

## Inspiration

At Taifa, we were initially focused on providing solar panel detection images. However, I saw an opportunity to enhance this service by detecting faulty cells within the panels. Inspired by the high costs and resource demands of existing market solutions, I proposed developing an in-house solution.

I created a proof of concept (POC) and presented it to my manager, highlighting its potential to save costs and improve efficiency. This POC led to the development of a fully functional product that is now in production, though it remains confidential.

## Overview

The goal of this POC is to demonstrate the feasibility of using a YOLO V3 CNN models to accurately identify and classify faulty solar panel cells from thermal drone images. This approach aims to improve the efficiency and reliability of solar panel maintenance and fault detection processes.

<p>
  <img src="https://github.com/Aakash1417/Solar-Panel-fault-detection-POC---public-view/blob/main/input/site_3_example.jpg" width="300" alt="Smaple thermal image"/>
  <img src="https://github.com/Aakash1417/Solar-Panel-fault-detection-POC---public-view/blob/main/output/site_3_example.jpg" width="300" alt="Annotated Smaple thermal image with fault cells marked"/> 
</p>

## Dataset

The training dataset used for this project is confidential and proprietary to Taifa Engineering. Due to the sensitive nature of the data, it is not included in this repository. The dataset comprises images of solar panels with various fault conditions, annotated for object detection tasks.

## Project Structure

The repository is structured as follows:

-   `callbacks.py`: Contains custom callback functions for the Keras model training process.
-   `yolo_config.json`: Configuration file specifying model parameters, training settings, and dataset paths.
-   `gen_anchors.py`: Script for generating anchor boxes based on the training dataset.
-   `generator.py`: Defines a data generator for training the model with batch processing.
-   `predict.py`: Script for performing predictions on new images using the trained model.
-   `train.py`: Main script for training the YOLO model on the solar panel fault detection dataset.
-   `utils/`: Directory containing utility functions and classes for data preprocessing, model evaluation, and other tasks.
-   `voc.py`: Script for parsing annotations in the VOC format.
-   `yolo.py`: Contains the definition of the YOLO model and related functions.

## Getting Started

To get started with this project, please ensure you have Python 3.x installed along with all the dependencies listed in `requirements.txt`. You can install the dependencies using the following command:

```sh
pip install -r requirements.txt
```

Training the Model
To train the model, run the train.py script with the path to the configuration file as an argument:

```sh
python train.py -c yolo_config.json
```
