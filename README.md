# TripleTen-Sprint-15-Project
🖼️ Sprint 15: Computer Vision — Age Prediction
📝 Project Overview

This project focuses on predicting the age of individuals from facial images using computer vision techniques. Before applying sophisticated models like convolutional neural networks (CNNs), the goal is to perform exploratory data analysis (EDA) to understand the dataset and prepare it for modeling.

The project emphasizes understanding data distributions, image quality, and variability across age groups, which is essential for building accurate predictive models.

📊 Dataset Description

The dataset was obtained from ChaLearn Looking at People and is located in the /datasets/faces/ folder. It contains:

final_files/ — 7.6k photos of faces

labels.csv — CSV file with labels, containing two columns:

file_name — name of the image file

real_age — the age of the person in the image

Notes:

Due to the large number of images, it is recommended to read images sequentially using ImageDataGenerator to avoid memory overload.

Images vary in age, facial expressions, and lighting conditions.

🎯 Objectives

Perform exploratory data analysis to understand the dataset

Analyze dataset size and structure

Explore the age distribution of the images

Display 10–15 sample photos across different ages to visually inspect the data

Build and train a CNN model for age prediction

Provide actionable insights for age estimation tasks

🔍 Key Tasks Performed
🧹 Data Preparation

Loaded labels.csv and inspected the structure

Verified file paths for all images in final_files/

Checked for missing or inconsistent labels

📈 Exploratory Analysis

Calculated total number of images and unique age entries

Plotted age distribution histogram to visualize dataset balance

Generated summary statistics: mean, median, min, max ages

🖼️ Image Inspection

Displayed 10–15 representative images across different age ranges

Observed variability in image quality, lighting, and facial expressions

Identified potential preprocessing requirements (e.g., resizing, normalization, augmentation)

🔧 Model Building and Training

After completing EDA, the next step is to build and train a CNN model. Typical functions to structure the workflow include:

load_train(path)       # loads the train dataset
load_test(path)        # loads the test dataset
create_model(input_shape)  # defines the CNN model architecture
train_model(model, train_data, test_data, batch_size, epochs, steps_per_epoch, validation_steps)  # trains the model


Important Note:
Training a CNN on this dataset requires a GPU for reasonable time. Since GPUs are no longer available on the platform, we used a pre-trained model that was trained over 20 epochs. The model outputs and predictions are provided for further analysis.

You can also download the dataset for local usage and train your own model if a GPU is available.

📊 Model Evaluation

Evaluated the pre-trained model’s predictions against validation data

Compared predicted ages with true ages using error metrics (e.g., MAE or RMSE)

Visualized predictions for a few sample images to inspect model performance

🛠️ Tools & Technologies

Python

Pandas & NumPy for data analysis

Matplotlib & Seaborn for visualization

TensorFlow / Keras for CNN modeling

ImageDataGenerator for sequential image loading and augmentation

Jupyter Notebook

📈 Final Deliverables

Dataset summary including total images and age distribution

Visualizations of age histogram and sample images

Pre-trained CNN model results for age prediction

Insights on dataset variability, image quality, and model performance

Ready-to-use dataset and code framework for further CNN training locally

✅ Success Criteria

A successful project demonstrates:

Clear understanding of dataset size and structure

Insightful analysis of age distribution and image variability

Ability to build a CNN model pipeline for age prediction

Use of pre-trained model outputs when GPU training is not possible

Well-documented, reproducible workflow

📌 Conclusion

This project illustrates the application of computer vision and CNNs in real-world age prediction tasks. By exploring the dataset, inspecting images, and training (or using) a CNN model, the project provides insights into model performance and prepares the workflow for future improvements.

The approach ensures efficient data handling and model deployment readiness, even when GPU resources are limited.
