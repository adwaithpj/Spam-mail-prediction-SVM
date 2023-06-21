<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <style>
    .title {
      font-size: 24px;
      color: #333;
      text-decoration: underline;
    }

    .section {
      margin-bottom: 20px;
    }

    .step {
      margin-left: 20px;
    }

    .emoji {
      font-size: 20px;
      margin-right: 5px;
    }

    .highlight {
      background-color: #f8f8f8;
      padding: 10px;
      border-radius: 5px;
    }
  </style>
</head>
<body>
  <h1 class="title">Spam Email Prediction using SVM Model</h1>
  <p>This repository contains a Python implementation of a spam email prediction system using the SVM (Support Vector Machine) model. The SVM model is trained to classify emails as either spam or non-spam (ham) based on their content.</p>

  <div class="section">
    <h2 class="title">Dataset</h2>
    <p>The dataset used for training and testing the model is stored in the <code>spam_ham_dataset.csv</code> file. It contains email samples labeled as either "spam" or "ham" (non-spam). The dataset is loaded into a pandas DataFrame for further processing.</p>
  </div>

  <div class="section">
    <h2 class="title">Data Preprocessing</h2>
    <p>The <code>spam_prediction.ipynb</code> Jupyter notebook is used for data preprocessing, including filtering and text feature extraction. The following steps are performed:</p>
    <ol>
      <li class="step">
        <span class="emoji">⚙️</span>
        The dataset is loaded into a pandas DataFrame, and any null values are replaced with an empty string.
      </li>
      <li class="step">
        <span class="emoji">⚙️</span>
        The emails are separated into the text (X) and label (Y) data.
      </li>
      <li class="step">
        <span class="emoji">⚙️</span>
        The data is split into training and testing sets using an 80-20 train-test split.
      </li>
      <li class="step">
        <span class="emoji">⚙️</span>
        The text data is transformed into feature vectors using the TfidfVectorizer, which converts the text to lowercase and removes English stop words.
      </li>
      <li class="step">
        <span class="emoji">⚙️</span>
        The label values are converted to integers for model training.
      </li>
    </ol>
  </div>

  <div class="section">
    <h2 class="title">Model Training</h2>
    <p>The SVM model is trained using the LinearSVC class from scikit-learn. The training data (X_train_features and Y_train) is used to train the model. The accuracy of the model is evaluated using both the training data and the testing data. The model achieves a training accuracy of 100% and a testing accuracy of 98%.</p>
  </div>

  <div class="section">
    <h2 class="title">Prediction on New Emails</h2>
    <p>The <code>spam_prediction_main.py</code> script contains the program for predicting whether a new email is spam or non-spam (ham). The user is provided with a menu to choose the type of input for prediction: subject only, whole email, or exit.</p>
    <ol>
      <li class="step">
        <span class="emoji">📨</span>
        If the user chooses the subject option, they need to enter the subject of the email.
      </li>
      <li class="step">
        <span class="emoji">📨</span>
        If the user chooses the whole email option, they need to enter the entire text of the email.
      </li>
      <li class="step">
        <span class="emoji">🔍</span>
        The input is then transformed into feature vectors using the same TfidfVectorizer used during training.
      </li>
      <li class="step">
        <span class="emoji">🤖</span>
        The trained SVM model predicts whether the email is spam or non-spam based on the input.
      </li>
      <li class="step">
        <span class="emoji">💡</span>
        The prediction result is displayed as "SPAM" or "Inbox Mail."
      </li>
    </ol>
  </div>

  <div class="section">
    <h2 class="title">Usage</h2>
    <p>To use this spam email prediction system, follow these steps:</p>
    <ol>
      <li class="step">
        <span class="emoji">🔧</span>
        Ensure that Python and the required dependencies (numpy, pandas, scikit-learn) are installed.
      </li>
      <li class="step">
        <span class="emoji">📥</span>
        Clone or download this repository.
      </li>
      <li class="step">
        <span class="emoji">▶️</span>
        Run the <code>spam_prediction_main.py</code> script.
      </li>
      <li class="step">
        <span class="emoji">📝</span>
        Choose the appropriate option to enter the subject or whole text of the email for prediction.
      </li>
      <li class="step">
        <span class="emoji">🔄</span>
        Repeat the process for multiple email predictions.
      </li>
      <li class="step">
        <span class="emoji">🚪</span>
        To exit the application, choose the exit option from the menu.
      </li>
    </ol>
  </div>

  <div class="section">
    <h2 class="title">Conclusion</h2>
    <p>This spam email prediction system utilizes the SVM model to classify emails as spam or non-spam. It achieves high accuracy on both the training and testing data, making it effective in identifying spam emails. By following the provided steps, users can easily utilize the system to predict the classification of new emails.</p>
  </div>

  <div class="section">
    <h2 class="title">License</h2>
    <p>This project is licensed under the <a href="LICENSE">MIT License</a>.</p>
  </div>
</body>
</html>
