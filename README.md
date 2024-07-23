# Solar Panel Fault Detection - Proof of Concept (POC)

## Overview

The goal of this POC is to demonstrate the feasibility of using a YOLO V3 CNN models to accurately identify and classify faulty solar panel cells from thermal drone images. This approach aims to improve the efficiency and reliability of solar panel maintenance and fault detection processes. Checkout the outputs detected here: [outputs](https://github.com/Aakash1417/Solar-Panel-fault-detection-POC---public-view/blob/main/output/)

The CNN architecture here is based on the YOLOv3 model, which is designed for object detection tasks. It uses a series of convolutional layers with batch normalization and leaky ReLU activations, organized in blocks with skip connections [1]. The architecture includes three detection heads at different scales, which helps in detecting objects of various sizes effectively. I chose to built the architecture this way because:

-   Multi-Scale Detection: By having three different detection heads, the model can capture features at different resolutions, improving its ability to detect small and large objects.

-   Skip Connections: These connections help in retaining spatial information and making the training of deeper networks more stable.

-   Efficient Convolutional Blocks: The use of batch normalization and leaky ReLU improves the training speed and accuracy by normalizing activations and adding non-linearity.

-   Custom YOLO Layer: This layer processes the output to match the format required for calculating the loss and making predictions, handling the complexity of bounding box prediction and class probabilities.

## Limitations

As of now, there are certain limitations in the approach:

-   Overlapping Cells: There can be overlap, especially if there are overlapping cells in the set of images. Therefore, all images must be hand-picked to avoid overlap.
-   Detection Precision: Not all defects are detected, only some. The model achieves a precision score of around ~72%.

<p>
  <img src="https://github.com/Aakash1417/Solar-Panel-fault-detection-POC---public-view/blob/main/input/site_1_image_32.jpg" width="300" alt="Sample thermal image"/>
  <img src="https://github.com/Aakash1417/Solar-Panel-fault-detection-POC---public-view/blob/main/output/site_1_image_32.jpg" width="300" alt="Annotated Sample thermal image with fault cells marked"/> 
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

## References

-   [1] [Solar panel defect detection design based on YOLO v5 algorithm](https://www.sciencedirect.com/science/article/pii/S2405844023060346)
-   [2] [A multi-stage model based on YOLOv3 for defect detection in PV panels based on IR and visible imaging by unmanned aerial vehicle](https://www.sciencedirect.com/science/article/abs/pii/S0960148122005079)
