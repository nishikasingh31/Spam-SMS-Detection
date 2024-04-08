# Spam SMS Detection

This repository contains Python code for building an AI model that can classify SMS messages as spam or legitimate (ham). The model is built using techniques like TF-IDF or word embeddings with classifiers like Naive Bayes, Logistic Regression, or Support Vector Machines to identify spam messages.

## Dataset

The dataset used for this project is the SMS Spam Collection dataset. It consists of SMS messages in English, with a total of 5,574 messages tagged as spam or ham (legitimate).

## Code Overview

- `spam_sms_detection.ipynb`: Jupyter Notebook containing the Python code for building the spam SMS detection model.
- `spam.csv`: CSV file containing the SMS dataset.

## Dependencies

- pandas
- scikit-learn
- matplotlib
- numpy

You can install the dependencies using the following command:

pip install -r requirements.txt

## Usage

1. Clone the repository:

git clone https://github.com/nishikasingh31/spam-sms-detection.git

2. Navigate to the repository directory:

cd spam-sms-detection

3. Install the dependencies:

pip install -r requirements.txt

4. Run the Jupyter Notebook `spam_sms_detection.ipynb` to train and evaluate the spam SMS detection model.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

