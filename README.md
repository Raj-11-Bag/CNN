# Oxford-IIIT Pet Dataset Classification using CNN 

This project focuses on classifying pet breeds using the Oxford-IIIT Pet dataset. It involves data preprocessing, defining a Convolutional Neural Network (CNN) model, training the model, and evaluating its performance.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Data Overview](#data-overview)
- [Usage](#usage)
- [Visualizations](#visualizations)
- [Model Evaluation](#model-evaluation)
- [Contributing](#contributing)
- [License](#license)

## Introduction
The Oxford-IIIT Pet Dataset Classification project demonstrates the use of TensorFlow and TensorFlow Datasets to build a CNN model for classifying images of pets into different breeds. The project covers:
- Loading and preprocessing the dataset
- Building and training a CNN model
- Evaluating the model's performance
- Visualizing the training process and results

## Installation
To run this project locally, you need to have Python and the required libraries installed. Follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/oxford-pet-classification.git
    cd oxford-pet-classification
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Data Overview
The Oxford-IIIT Pet dataset contains images of 37 breeds of pets, with roughly 200 images per breed. Each image includes a label indicating the breed of the pet.

### Sample Data
Here's a brief overview of the dataset:

| Attribute     | Description                    |
|---------------|--------------------------------|
| Image         | An image of a pet              |
| Label         | The breed of the pet (0-36)    |

Below is a sample of the dataset:

| Image                                   | Label |
|-----------------------------------------|-------|
| ![Sample Image 1](path_to_sample_image1) | 5     |
| ![Sample Image 2](path_to_sample_image2) | 12    |
| ![Sample Image 3](path_to_sample_image3) | 7     |

## Usage
To run the project, you can use either Jupyter Notebook or a Python script. Below are the instructions for both:

### Running the Jupyter Notebook
1. Start Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

2. Open the `oxford_pet_classification.ipynb` notebook.
3. Run the cells in the notebook to execute the data preprocessing, model training, and evaluation steps.

### Running the Python Script
1. Ensure you have the `oxford_pet_classification.py` script in the project directory.
2. Run the script:
    ```bash
    python oxford_pet_classification.py
    ```

Ensure that the dataset is correctly downloaded and accessible by TensorFlow Datasets.

## Visualizations
This project includes visualizations to understand the dataset and the model's performance. These visualizations help in analyzing the data distribution and the model's training progress.

### Dataset Visualizations
- **Sample Images**: Display sample images from the Oxford-IIIT Pet dataset to get an idea of the data.
- **Label Distribution**: Plot the distribution of different pet breeds in the dataset.

### Training Visualizations
- **Training Accuracy and Loss**: Plot the training and validation accuracy and loss over epochs to monitor the model's performance.

## Model Evaluation
The project involves training a Convolutional Neural Network (CNN) on the Oxford-IIIT Pet dataset and evaluating its performance.

### Steps Involved:
1. **Data Loading and Preprocessing**:
   - Load the dataset using `tensorflow_datasets`.
   - Resize images to 128x128 pixels.
   - Normalize the pixel values to range between 0 and 1.

2. **Data Splitting**:
   - Split the dataset into training and testing sets (70% training, 30% testing).

3. **Model Definition**:
   - Define a simple CNN model with three convolutional layers followed by max-pooling layers, a flatten layer, and two dense layers.

4. **Model Compilation**:
   - Compile the model using the Adam optimizer and sparse categorical cross-entropy loss function.
   - Use accuracy as the evaluation metric.

5. **Model Training**:
   - Train the model on the training dataset for 10 epochs.
   - Validate the model on the testing dataset.

6. **Model Evaluation**:
   - Evaluate the model's performance on the testing dataset.
   - Print the test loss and test accuracy.

### Model Summary:
The model summary provides detailed information about each layer, including the output shape and the number of parameters. Below is a brief summary:

- **Conv2D Layers**: Extract spatial features from the images.
- **MaxPooling2D Layers**: Downsample the feature maps to reduce the computational load.
- **Flatten Layer**: Convert the 2D feature maps into a 1D feature vector.
- **Dense Layers**: Perform the final classification into 37 pet breeds.

## Contributing
Contributions to this project are welcome! If you would like to contribute, please follow these steps:

1. Fork the repository and clone it to your local machine.
2. Create a new branch for your feature or bug fix.
3. Make your changes and test thoroughly.
4. Commit your changes with descriptive commit messages.
5. Push your changes to your forked repository.
6. Open a pull request against the `main` branch of this repository.

Please ensure your contributions adhere to the coding standards and focus on enhancing the project's functionality or resolving existing issues.

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.
