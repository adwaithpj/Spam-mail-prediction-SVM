import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
import json


def execute_notebook(notebook_path, input_variable):
    # Read the notebook file
    with open(notebook_path) as f:
        nb = nbf.read(f, as_version=4)

    # Modify the input cell
    input_cell_index = 23  # Index of the input cell in the notebook
    nb.cells[input_cell_index].source = f"input_value = '{input_variable}'"

    # Execute the notebook
    ep = ExecutePreprocessor(timeout=600)
    ep.preprocess(nb, {'metadata': {'path': './'}})

    # Extract the outputs from executed cells
    outputs = []
    for cell in nb.cells:
        if 'outputs' in cell:
            for output in cell['outputs']:
                if 'text' in output:
                    outputs.append(output['text'].strip())

    return outputs

def send_data():
    notebook_path = 'spam_prediction.ipynb'
    input_value = input("Enter the Subject of mail :")
    result_outputs = execute_notebook(notebook_path, input_value)
    for output in result_outputs:
        print(output)

    
        
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
        send_data()
    if(x==2):
        send_data()
    if(x==3):
        cond=0
    else:
        print("Please enter a valid Option")
        


# input_value1 = input("Enter the mail Subject")

