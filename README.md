# Titanic Survival Prediction

A classic machine learning project to predict passenger survival on the Titanic, based on the Kaggle dataset.

---

## 🌟 Project Overview

This repository contains a complete data science workflow for predicting survival on the Titanic. The project tackles a supervised classification problem by analyzing a dataset of passenger information and using it to train a machine learning model. The goal is to build a model that can accurately predict whether a passenger survived or not, given their characteristics such as age, gender, and class.

The workflow includes:

- Data Cleaning and Exploration: Handling missing values, identifying key features, and visualizing the data to understand the relationships between variables and survival.

- Feature Engineering: Creating new, more informative features from the existing data to improve model performance.

- Model Training and Evaluation: Building and training a classification model and evaluating its performance using metrics like accuracy, precision, and recall.

---

## 📊 Dataset

Source link: https://www.kaggle.com/competitions/titanic/data

The project utilizes the Titanic Survival Prediction dataset, which is publicly available on Kaggle. It is typically split into two files:

- train.csv: Contains the training data with the survival outcome for each passenger.

- test.csv: Contains the testing data without the survival outcome, used for making predictions.

The dataset includes a variety of features, such as:

- PassengerId: A unique ID for each passenger.

- Survived: The target variable, indicating survival (0 = No, 1 = Yes).

- Pclass: The passenger's ticket class (1, 2, or 3).

- Name: The passenger's name.

- Sex: The passenger's gender.

- Age: The passenger's age in years.

- SibSp: Number of siblings/spouses aboard.

- Parch: Number of parents/children aboard.

- Ticket: The ticket number.

- Fare: The passenger fare.

- Cabin: The cabin number.

- Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

---

## 🛠️ Technologies Used

- Python: The primary programming language for the project.

- Pandas: Essential for data manipulation and analysis.

- NumPy: Used for numerical operations and array handling.

- Scikit-learn: The main library for building and evaluating the machine learning models.

- Matplotlib & Seaborn: Used for data visualization and exploratory data analysis.

- TensorFlow & Keras: The deep learning frameworks used to create and train the MLP model.

- Jupyter Notebooks: A common environment for developing the project and documenting the process.

- Git & GitHub: Used for version control and project hosting.

---

## 🧠 Deep Learning Model: Multi-Layer Perceptron (MLP) Features

The MLP model introduces advanced capabilities for improved prediction accuracy. Key features of this implementation include:

- Multiple Hidden Layers: The network can be configured with multiple hidden layers to capture complex, non-linear relationships in the data.

- Optimizers: Utilize modern optimization algorithms like Adam to efficiently train the network and speed up convergence.

- Activation Functions: Each neuron in the hidden layers uses non-linear activation functions such as ReLU (Rectified Linear Unit) to introduce non-linearity and enable the model to learn complex patterns.

- Regularization: Implements techniques like Dropout to prevent overfitting, ensuring the model generalizes well to new, unseen data.

- Cross-Entropy Loss: Uses a cross-entropy loss function, which is ideal for binary classification problems like predicting survival.
