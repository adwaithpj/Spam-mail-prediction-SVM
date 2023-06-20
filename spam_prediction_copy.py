
import numpy as np 
import pandas as pd

#ML
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

#Accuracy Metrics
from sklearn.metrics import accuracy_score
#Data Pre Processing
#load the dataset to pandas Data Frame
raw_mail_data = pd.read_csv('spam_ham_dataset.csv') #read the data in csv to pandas data frame

#replace the null values with a null string
#store it in a variable called mail_data
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')
mail_data.shape
mail_data.head()

# #label spam mail as 0; and non-spam mail as 1
# mail_data.loc[mail_data['label'] == 'spam','Category',]= 0
# mail_data.loc[mail_data['label'] == 'ham','Category',]=1# mail_data.head()# 0 -> ham , 1-> spam
#seperate the data as text and label
# X-> test , Y -> label


X = mail_data['text']
Y = mail_data['label_num']
print(X)
print(Y)
#Train Test Split

# split the data as train data and test data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size = 0.8,test_size = 0.2, random_state =3)

#Feature Extraction

#transform the text data to feature vectors that can be used as input to the SVM model using TfidVectorizer
#convert the text to lower case letters

feature_extraction = TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

#conver Y_train and Y_test values as integers 

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

#Training the model -> Support Vector Machine#training the support vector machine with training data

model = LinearSVC()
model.fit(X_train_features,Y_train)

#Evaluation of the model
#prediction on training data 

predicton_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train,predicton_on_training_data)
print("Accuracy on training data : ",accuracy_on_training_data)

#prediction on test data 
predicton_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test,predicton_on_test_data)
print("Accuracy on Test data : ",accuracy_on_test_data)

#Prediction on new mail

# input_mail = ["looking for medication ? we ` re the best source "]
# #convert text to feature vectors
# input_mail_features = feature_extraction.transform(input_mail)

# #making prediction

# prediction = model.predict(input_mail_features)
# print(prediction)
# if prediction == 1:
#     print("SPAM")
# else:
#     print("Inbox Mail")input_value = get_ipython().getoutput('python -c "import spam_predictio_main; print(spam_predictio_main.input_variable)"')[0]


#convert text to feature vectors
input_main = 
input_mail_features = feature_extraction.transform(input_mail)

#making prediction

prediction = model.predict(input_mail_features)
print(prediction)
if prediction == 1:
    print("SPAM")
else:
    print("Inbox Mail")