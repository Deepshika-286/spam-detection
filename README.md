# Spam Email detection

This project implements a Spam Email Classifier that uses a Naive Bayes model to classify emails into spam or ham (non-spam). It follows a structured approach for data collection, preprocessing, model creation, training, and evaluation.

## Libraries Used
- pandas: For loading and manipulating the dataset.

- numpy: For numerical operations.

- scikit-learn: For machine learning, including the Naive Bayes classifier and text vectorization.

- gensim: For text preprocessing and stemming.

- chardet: For detecting file encoding.

## Workflow
- Data Collection: The dataset used is spam.csv, which contains labeled emails (spam and ham). It is loaded into a pandas DataFrame for processing.
- Data Preprocessing:
Detecting and handling encoding issues with chardet, Mapping the target labels to binary values (0 for spam and 1 for ham), Lowercasing the text and applying stemming, Using CountVectorizer to convert the text into numerical feature vectors.
- Model Creation: A Naive Bayes classifier (MultinomialNB) is created using scikit-learn. The model is trained on the preprocessed text data.
- Model Training: The model is trained on a training dataset and then evaluated on a test dataset to predict whether emails are spam or ham.
- Model Evaluation: The model is evaluated using classification metrics such as Precision, Recall, F1-score, Accuracy
- Spam Detection: The trained model can be used to predict whether new emails are spam or ham by calling the prediction function.

## Setup and Usage
- Clone the Repository

      git clone https://github.com/yourusername/spam-email-classification.git
      cd spam-email-classification
- Install Dependencies

      pip install -r requirements.txt

Alternatively, you can install the libraries manually:

    pip install pandas numpy scikit-learn gensim chardet
- Run the Script: Ensure the dataset spam.csv is available in the project directory. Then, run the Python script to train and evaluate the model.

- Make Predictions Use the predict() function to classify new email texts as spam or ham:
## Contributions
Feel free to fork this repository, make improvements, or add new features to enhance the translation capabilities. Contributions are welcome!
### Hope this helps! Happy Learning!!
