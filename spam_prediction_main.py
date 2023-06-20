import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
import json

#assigning jupyter notebook path
notebook_path = 'spam_prediction.ipynb'
notebook = nbf.read(notebook_path, as_version=4)

def send_data():
    input_values = input("Enter the Subject of mail :")
    
#input values 
cond=1
while(cond==1):
    print("To check whether a mail is 'spam' or 'ham'\n")
    print("****************************\n")
    print("1. Enter the Subject of Mail\n")
    print("2. Enter the Whole text \n")
    print("3. Exit the application\n")
    try:
        x = int(input("Enter the choice : "))
        print("You entered:", x)
    except ValueError:
        print("Invalid input. Please enter an integer.")

    if(x==1):
        result = 
        


input_value1 = input("Enter the mail Subject")
input_value2 = 
