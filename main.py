import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import functions
import models
import json
import os

df = None
prev_df = None #Last version of the updated DataFrame
X_train = None
X_test = None
y_train = None
y_test = None

#Loading the CSV file
def load_file():
    global df
    df = functions.load_csv_file()
    message_bar_main.config(text="File loaded successfully! \nClick on Load CSV File again to load another file. \nClick on the View tab any time to see the current state of your DataFrame.")

#Viewing the file
def view_file():
    global df
    functions.display_data(df)

#Updating prev_df for undo and redo functions
def update_prev_df(function):
    global prev_df
    global df
    temp_df = df.copy()
    df = function(df)
    if temp_df.equals(df):
        return
    else:
        prev_df = temp_df

#Check if prev_df needs to be updated   
def find_change(df, temp_df):
    if df.equals(temp_df):
        return False
    else:
        return True

#Columns dropping functionality    
def drop_column():
    global df
    temp_df = df.copy()

    update_prev_df(functions.drop_column)
    message_bar_preprocessing.pack(pady=25)
    if find_change(df, temp_df):
        message_bar_preprocessing.config(text="Column(s) dropped!")
    else:
        message_bar_preprocessing.config(text="Couldn't drop the column(s)")

#Functionality to specify ordinal columns
def specify_ordinal_columns():
    global df
    temp_df = df.copy()
    update_prev_df(functions.specify_ordinal_columns)
    message_bar_preprocessing.pack(pady=25)
    if find_change(df, temp_df):
        message_bar_preprocessing.config(text="Ordinal columns specified!")
    else:
        message_bar_preprocessing.config(text="Couldn't specify ordinal columns")

#Functionality to handle Null values
def handle_missing_values():
    global df
    temp_df = df.copy()
    update_prev_df(functions.handle_missing_values)

    message_bar_preprocessing.pack(pady=25)
    if find_change(df, temp_df):
        message_bar_preprocessing.config(text="Handled Null Values")
    else:
        message_bar_preprocessing.config(text="Couldn't complete handling")

#Temporary popup window to display any messages
def show_popup_message(message):
    popup = tk.Toplevel(root)
    popup.geometry("150x35+1000+600")
    popup.overrideredirect(True)

    # Label inside the popup
    label = tk.Label(popup, text=message, bg="gray", fg="yellow")
    label.pack(padx=10, pady=10)
    popup.after(2000, popup.destroy)

#Undo functionality
def undo():
    global df
    global prev_df

    if prev_df is None or prev_df.empty:
        return
    temp_df = df.copy()
    df = prev_df.copy()
    prev_df = temp_df
    
    show_popup_message("Undo action completed!")

#Redo functionality
def redo():
    global df
    global prev_df

    if prev_df is None or prev_df.empty:
        return
    
    temp_df = df.copy()
    df = prev_df.copy()
    prev_df = temp_df
    
    show_popup_message("Redo action completed!")

#Data Filtering process
def filter_data():
    global df
    temp_df = df.copy()
    update_prev_df(functions.filter_data)
    message_bar_train.pack(pady=20)
    if find_change(df, temp_df):
        message_bar_train.config(text="Data Filtered!")
    else:
        message_bar_train.config(text="Filtering not possible")

#Data Splitting Process
def split_data():
    global df, X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = models.split_data(df)

def save_model_results(file_name, classification_results=None, regression_results=None):
    # Check if the file already exists
    if os.path.isfile(file_name) and os.path.getsize(file_name) > 0:
        # Load existing data
        with open(file_name, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = {
                    "classification_models": [],
                    "regression_models": []
                }
    else:
        # Create new structure if file doesn't exist
        data = {
            "classification_models": [],
            "regression_models": []
        }

    # Add or update classification model results
    if classification_results:
        for model_name, metrics in classification_results.items():
            accuracy, f1_score, confusion_matrix = metrics

            # Convert confusion matrix to list if it's a NumPy array
            if isinstance(confusion_matrix, np.ndarray):
                confusion_matrix = confusion_matrix.tolist()

            # Check if model exists, and update or append
            existing_model = next((item for item in data["classification_models"] if item["model_name"] == model_name), None)
            if existing_model:
                existing_model.update({
                    "accuracy": accuracy,
                    "f1_score": f1_score,
                    "confusion_matrix": confusion_matrix
                })
            else:
                data["classification_models"].append({
                    "model_name": model_name,
                    "accuracy": accuracy,
                    "f1_score": f1_score,
                    "confusion_matrix": confusion_matrix
                })

    # Add or update regression model results (handling dictionary input)
    if regression_results:
        for model_name, metrics in regression_results.items():
            mse, rsquared, mae = metrics

            # Check if model exists, and update or append
            existing_model = next((item for item in data["regression_models"] if item["model_name"] == model_name), None)
            if existing_model:
                existing_model.update({
                    "mse": mse,
                    "r2_score": rsquared,
                    "mae": mae
                })
            else:
                data["regression_models"].append({
                    "model_name": model_name,
                    "mse": mse,
                    "r2_score": rsquared,
                    "mae": mae
                })

    # Save or overwrite the file with updated results
    with open(file_name, 'w') as file:
        json.dump(data, file, indent=4)

    message_bar_train.config(text="Results saved!")

#Popup to display model results, containing the save button as well
def display_classifier_results(model_name, classification_results):

    accuracy_score = classification_results[model_name][0]
    f1_score = classification_results[model_name][1]
    confusion_matrix = classification_results[model_name][2]

    popup = tk.Toplevel()
    popup.geometry("300x180")
    popup.title(f"{model_name} Results")
    results = tk.Label(popup, text=f"Accuracy Score: {accuracy_score} \nF1 Score: {f1_score} \nConfusion Matrix: \n{confusion_matrix}")
    results.pack(pady=10, padx=10)

    save_button = tk.Button(popup, text="Save Results", command=lambda: save_model_results("results.json", classification_results = classification_results, regression_results=None))
    save_button.pack(pady=10)

#Popup to display model results, containing the save button as well
def display_regressor_results(model_name, regression_results):

    mse = regression_results[model_name][0]
    rsquared = regression_results[model_name][1]
    mae = regression_results[model_name][2]

    popup = tk.Toplevel()
    popup.geometry("300x180")
    popup.title(f"{model_name} Results")
    results = tk.Label(popup, text=f"Mean Squared Error: {mse} \nR-Squared: {rsquared} \nMean Absolute Error: {mae}")
    results.pack(pady=10, padx=10)

    save_button = tk.Button(popup, text="Save Results", command=lambda: save_model_results("results.json", regression_results = regression_results, classification_results=None))
    save_button.pack(pady=10)

#Choose model to train on
def select_model():
    global df, X_train, X_test, y_train, y_test
    user_choice = functions.popup_input("Choose your model type: \n1. Classification Model \n2. Regression Model")        
    message_bar_train.pack(pady=20)
    if user_choice is None or user_choice == "":
        print("Action cancelled by the user")
        return

    while user_choice != "1" and user_choice != "2":
        user_choice = functions.popup_input("Please enter a valid input, '1' for Classifical Model and '2' for Regression Model:")

    if user_choice == "1":
        user_choice = functions.popup_input("Choose one of the following models: \n1. Random Forest Classifier \n2. XGBoost Classifier \n3. SVM Classifier \n\nEnter your choice: ")
        match user_choice:
            case "1":
                accuracy_score, f1_score, confusion_matrix = models.RFC(X_train,X_test,y_train,y_test)
                classification_results = {"Random Forest Classifier" : (accuracy_score, f1_score, confusion_matrix)}
                display_classifier_results("Random Forest Classifier", classification_results)
                message_bar_train.config(text="Random Forest Classifier trained!")

            case "2":
                accuracy_score, f1_score, confusion_matrix = models.XGBC(X_train,X_test,y_train,y_test)
                classification_results = {"XGBoost Classifier" : (accuracy_score, f1_score, confusion_matrix)}
                display_classifier_results("XGBoost Classifier", classification_results)
                message_bar_train.config(text="XGBoost Classifier trained!")
            case "3":
                accuracy_score, f1_score, confusion_matrix = models.SVMC(X_train,X_test,y_train,y_test)
                classification_results = {"SVM Classifier" : (accuracy_score, f1_score, confusion_matrix)}
                display_classifier_results("SVM Classifier", classification_results)
                message_bar_train.config(text="SVM Classifier trained!")

    elif user_choice == "2":
        user_choice = functions.popup_input("Choose one of the following models: \n1. Random Forest Regressor \n2. XGBoost Regressor \n3. Decision Tree Regressor \n\nEnter your choice: ")
        match user_choice:
            case "1":
                mse, rsquared, mae = models.RFR(X_train,X_test,y_train,y_test)
                regression_results = {"Random Forest Regressor" : (mse, rsquared, mae)}
                display_regressor_results("Random Forest Regressor", regression_results)
                message_bar_train.config(text="Random Forest Regressor trained!")
                
            case "2":
                mse, rsquared, mae = models.XGBR(X_train,X_test,y_train,y_test)
                regression_results = {"XGBoost Regressor" : (mse, rsquared, mae)}
                display_regressor_results("XGBoost Regressor", regression_results)
                message_bar_train.config(text="XGBoost Regressor trained!")
            case "3":
                mse, rsquared, mae = models.DTR(X_train,X_test,y_train,y_test)
                regression_results = {"Decision Tree Regressor" : (mse, rsquared, mae)}
                display_regressor_results("Decision Tree Regressor", regression_results)
                message_bar_train.config(text="Decision Tree Regressor trained!")

#Close all frames to display another frame
def close_all_frames():
    for widget in root.winfo_children():  # Iterate over all widgets in the root
        if isinstance(widget, tk.Frame):  # If the widget is a frame
            widget.pack_forget()          # Hide it (use grid_forget() if using grid)

#To show home tab
def show_main_frame():
    close_all_frames()
    main_frame.pack(fill="both", expand=True)

#To show preprocessing tab
def show_preprocessing_panel():
    close_all_frames()
    preprocessing_frame.pack(fill="both", expand=True)

#To show train tab
def show_train_frame():
    close_all_frames()
    train_frame.pack(fill="both", expand=True)


#Main application window
root = tk.Tk()
root.title("AutoML Solution")
root.geometry("1100x500")

#Menu bar
menu_bar = tk.Menu(root)
menu_bar.add_command(label="Home", command=show_main_frame)
menu_bar.add_command(label="View", command=view_file)
menu_bar.add_command(label="Preprocessing", command=show_preprocessing_panel)
menu_bar.add_command(label="Train", command=show_train_frame)
menu_bar.add_command(label="Undo", command=undo)
menu_bar.add_command(label="Redo", command=redo)

#Main frame
main_frame = tk.Frame(root, bg="#065c20")
main_frame.pack(fill="both", expand=True)

#Main screen message
welcome_label = tk.Label(main_frame, text="Auto ML for tabular data", font=("Helvetica", 24), bg="green", fg="white")
welcome_label.pack(pady=20)

#Load button
load_button = tk.Button(main_frame, text="Load CSV File", command=load_file)
load_button.pack(pady=20)

#Message bar for announcements
message_bar_main = tk.Label(main_frame, text="", font=("Helvetica", 12), bg="green", fg="yellow")
message_bar_main.pack(pady=20)

# Preprocessing frame
preprocessing_frame = tk.Frame(root, bg="#0a1f6e")

# Preprocessing frame main text
preprocessing_label = tk.Label(preprocessing_frame, text="Modify your dataset", font=("Helvetica", 24), bg="blue", fg="white")
preprocessing_label.pack(pady=20)

# Create Drop Columns button
drop_columns_button = tk.Button(preprocessing_frame, text="Drop Columns", command= drop_column, width=20, height=1)
drop_columns_button.pack(pady=(30,0), padx=30, anchor='w')

drop_columns_label = tk.Label(preprocessing_frame, text="Drop unnecessary columns from your dataset", font=("Helvetica", 12), bg="blue", fg="white")
drop_columns_label.pack(pady=(0,30))

# Create Specify Ordinal Columns button
specify_ordinal_columns_button = tk.Button(preprocessing_frame, text="Specify Ordinal Columns", command= specify_ordinal_columns, width=20, height=1)
specify_ordinal_columns_button.pack(pady=(30,0), padx=30, anchor='w')

specify_columns_label = tk.Label(preprocessing_frame, text="Specify the categorical nature of your dataset", font=("Helvetica", 12), bg="blue", fg="white")
specify_columns_label.pack(pady=(0,30))

# Handle Missing Values Button
handle_missing_values_button = tk.Button(preprocessing_frame, text="Handle Missing Values", command= handle_missing_values, width=20, height=1)
handle_missing_values_button.pack(pady=(30,0), padx=30, anchor='w')

handle_missing_values_label = tk.Label(preprocessing_frame, text="Remove Null values from your dataset", font=("Helvetica", 12), bg="blue", fg="white")
handle_missing_values_label.pack(pady=(0,30))

# Message bar for announcements
message_bar_preprocessing = tk.Label(preprocessing_frame, text="", font=("Helvetica", 12), bg="blue", fg="yellow")

#Train frame
train_frame = tk.Frame(root, bg="#7f1c17")
#Train frame main text
train_label = tk.Label(train_frame, text="Train on your own model", font=("Helvetica", 24), bg="#d93131", fg="white")
train_label.pack(pady=20)

# Filter Data Button
filter_data_button = tk.Button(train_frame, text="Filter data", command= filter_data, width=20, height=1)
filter_data_button.pack(pady=(10,0), padx=30, anchor='w')
filter_data_label = tk.Label(train_frame, text="Slice your data with a specific criteria", font=("Helvetica", 12), bg="#d93131", fg="white")
filter_data_label.pack(pady=(0,10))

## Split Data Button
split_data_button = tk.Button(train_frame, text="Split data", command= split_data, width=20, height=1)
split_data_button.pack(pady=(10,0), padx=30, anchor='w')
split_data_label = tk.Label(train_frame, text="Split data into train and test", font=("Helvetica", 12), bg="#d93131", fg="white")
split_data_label.pack(pady=(0,10))

# Viewing train dataset
view_train_data_button = tk.Button(train_frame, text="View Training data", command= lambda: functions.display_data(pd.concat([X_train,y_train], axis=1)), width=20, height=1)
view_train_data_button.pack(pady=(10,0), padx=30, anchor='w')
view_train_data_label = tk.Label(train_frame, text="Once sliced, view your training dataset", font=("Helvetica", 12), bg="#d93131", fg="white")
view_train_data_label.pack(pady=(0,10))

# Viewing test dataset
view_test_data_button = tk.Button(train_frame, text="View Test data", command= lambda: functions.display_data(pd.concat([X_test,y_test], axis=1)), width=20, height=1)
view_test_data_button.pack(pady=(10,0), padx=30, anchor='w')
view_test_data_label = tk.Label(train_frame, text="Once sliced, view your test dataset", font=("Helvetica", 12), bg="#d93131", fg="white")
view_test_data_label.pack(pady=(0,10))

# Choose model button
model_button = tk.Button(train_frame, text="Select Model", command= select_model, width=20, height=1)
model_button.pack(pady=(10,0), padx=30, anchor='w')
model_label = tk.Label(train_frame, text="Choose your model and train (make sure that the data has been split)", font=("Helvetica", 12), bg="#d93131", fg="white")
model_label.pack(pady=(0,10))

#message window on train frame
message_bar_train = tk.Label(train_frame, text="", font=("Helvetica", 12), bg="#d93131", fg="yellow")

# Add the menu bar to the window
root.config(menu=menu_bar)
message_bar_main.config(text="Please proceed with loading your data to train your model on")

# Start the application
root.mainloop()
