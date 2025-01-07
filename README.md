# Fashion-Image-Classification-using-Pyspark-
Project Overview

This project demonstrates a scalable and efficient solution for classifying fashion images using Apache Spark's distributed computing capabilities. By leveraging PySpark, we accelerate the preprocessing, training, and evaluation of machine learning models on large-scale fashion datasets.
Features

    Distributed Data Processing: Utilizes PySpark for parallel computation to handle large-scale image datasets.
    Image Preprocessing: Efficient preprocessing with PySpark and OpenCV, including handling missing values, extracting image features, and converting data into feature vectors.
    Multi-Layer Perceptron Classifier: Implements an MLP model to classify fashion images based on features like season or clothing type.
    Evaluation Metrics: Calculates accuracy and displays a confusion matrix for classification performance analysis.
    Visualization: Uses Seaborn and Matplotlib to plot the confusion matrix for better insights.

Dataset

    Fashion Dataset: Images and metadata are sourced from Kaggle Fashion Product Images Dataset.
    The dataset includes images and their corresponding labels (e.g., season).

Installation

    Install the necessary Python libraries:

pip install pyspark matplotlib seaborn findspark pandas

Set up your Spark environment:

    Install Apache Spark and configure it for your system.
    Install Java (if not already installed).

Clone this repository and navigate to the project directory:

    git clone <repository_url>
    cd <repository_name>

Usage

    Update the image_dir and styles.csv paths in the script to point to your dataset's location.
    Run the Python script:

    python fashion_classification.py

Workflow

    Data Loading:
        Load images as a PySpark DataFrame.
        Read metadata (e.g., season labels) from a CSV file.

    Data Preprocessing:
        Handle missing values.
        Extract IDs from image filenames using regex.
        Transform labels into numerical representations.

    Feature Engineering:
        Convert raw image data into numerical arrays using UDFs.
        Assemble feature vectors for model training.

    Model Training:
        Use a Multi-Layer Perceptron (MLP) Classifier with defined layers for training.
        Split data into training (80%) and testing (20%) sets.

    Evaluation:
        Compute accuracy and generate a confusion matrix.
        Visualize results using Seaborn and Matplotlib.

Results

    Achieved a classification accuracy of 72.28%.
    Confusion matrix visualization highlights areas of improvement in prediction accuracy across classes.

Future Work
