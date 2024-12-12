Fake News Detection Project


Overview

This project aims to develop a machine learning model to detect fake news articles. The model uses a Passive Aggressive Classifier (PAC) algorithm, combined with TF-IDF vectorization, to classify news articles as either "FAKE" or "REAL".

Dataset

The dataset used for this project is a CSV file containing news articles labeled as either "FAKE" or "REAL". The dataset is split into training and testing sets using the train_test_split function from Scikit-learn.

Features

- TF-IDF Vectorization: The TF-IDF vectorizer is used to convert the text data into numerical features that can be fed into the machine learning model.
- Passive Aggressive Classifier (PAC): The PAC algorithm is used as the machine learning model to classify the news articles.
- Hyperparameter Tuning: The model's hyperparameters are tuned using the max_iter parameter to optimize the model's performance.

Results

The model achieves an accuracy of {round(score*100,2)}% on the test set. The confusion matrix is also provided to evaluate the model's performance.

Requirements

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- TensorFlow (optional)

Usage

1. Clone the repository using git clone.
2. Install the required libraries using pip install -r requirements.txt.
3. Run the  script to train and evaluate the model.

Contributing

Contributions are welcome! If you'd like to contribute to this project, please fork the repository and submit a pull request.

License

This project is licensed under the MIT License.
