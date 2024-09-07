import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import simpledialog
import pandas as pd

def load_csv_file():
    # Initialize Tkinter root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    
    # Open file dialog and allow user to select a file
    file_path = filedialog.askopenfilename(
        title="Select a CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    
    if file_path:
        try:
            # Load the CSV file into a pandas DataFrame
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            print(f"Error loading file: {e}")
    else:
        print("No file selected.")

def display_data(df):
    root = tk.Tk()
    root.title("Chosen CSV File")
    root.geometry("1000x1200")

    frame = tk.Frame(root)
    frame.pack(expand=True, fill='both')

    # Clear existing data
    for widget in frame.winfo_children():
        widget.destroy()

    if df is None or df.empty:
        error_label = tk.Label(frame, text="No file selected", font=("Helvetica", 16), fg="red")
        error_label.pack(expand=True)
    else:

        # Create Treeview
        tree = ttk.Treeview(frame, columns=list(df.columns), show='headings')
        tree.pack(expand=True, fill='both')

        # Define headings
        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, anchor="center")

        # Add rows
        for index, row in df.iterrows():
            tree.insert("", "end", values=list(row))

    root.mainloop()

def popup_input(prompt):
    # Create the main application window
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Make sure the root window is always on top
    root.attributes("-topmost", True)

    # Show the input dialog
    result = simpledialog.askstring("Input", prompt, parent=root)

    #destroy the root window
    root.destroy()

    return result

#try to convert to float
def convert_input(user_input):
    try:
        return float(user_input)
    except ValueError:
        return user_input


def drop_column(df):

    column_name = popup_input("Please enter the name(s) of the column(s) you want to drop. If there are multiple, please write their name comma-separated, without space:")
    columns = column_name.split(',')
    for name in columns:
        if not name in df.columns:
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            messagebox.showerror("Error", f"Column {name} not present")
            root.destroy()  # Destroy the hidden main window
            continue
        else:
            df.drop(name, axis=1, inplace=True)
            print(f"Given Column {name} dropped")

    return df

def specify_ordinal_columns(df):
    
    column_name = popup_input("Please enter the name(s) of the column(s) you want to specify as ordinal columns. If there are multiple, please write their name comma-separated, without space:")
    columns = column_name.split(',')
    for name in columns:
        if name not in df.columns:
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            messagebox.showerror("Error", f"Column {name} not present")
            root.destroy()  # Destroy the hidden main window
            continue
        else:
            order = simpledialog.askstring("Input", f"For the column {name}, please enter all the values in order. Separate them with a comma without a space:")
            order = order.split(',')
            df[name] = pd.Categorical(df[name], categories=order, ordered=True)
            print(f"Given Column {name} specified as ordinal")

    return df

def handle_missing_values(df):
    chosen_number = int(popup_input("Please choose a way to fill in for your missing values: \n1. Drop columns with null values \n2. Drop rows with null values \n3. Custom value \n\nEnter your number:"))
    match chosen_number:
        case 1:
            df = df.dropna(axis=1)
        case 2:
            df = df.dropna()
        case 3:
            value = popup_input("Please enter your value to replace all null values in the dataset: ")
            try:
                float(value)
                value = float(value)
            except ValueError:
                None
            
            df = df.fillna(value)

    return df

def filter_data(df):
    columns_list = list(df.columns)
    column_name = popup_input(f"Please enter the column you want to filter by: {columns_list}")
    if column_name not in columns_list:
        while(column_name not in columns_list):
            if column_name is None or column_name == "":
                print("Action canceled by the user.")
                return df  # Exit the loop if "Cancel" is clicked
            column_name = popup_input(f"Column name {column_name} not found in the columns. Please input one of these {columns_list}: ")
    
    comparison_operators_list = ["=","<",">","<=",">=","!="]
    comparison_operator = popup_input(f"Please choose the comparison operator for your column values: {comparison_operators_list} \n\nEnter your operator: ")
    if comparison_operator not in comparison_operators_list:
        while(comparison_operator not in comparison_operators_list):
            if comparison_operator is None or comparison_operator == "":
                print("Action canceled by the user.")
                return df  # Exit the loop if "Cancel" is clicked
            comparison_operator = popup_input(f"Comparison operator {comparison_operator} not found in the comparison operators list. Please input one of these {comparison_operators_list}:")
 
    criteria = popup_input(f"Please input the criteria: {column_name} {comparison_operator} ?:")
    criteria = convert_input(criteria)

    match comparison_operator:
        case "=":
            df = df[df[column_name] == criteria]
            return df
        case "<":
            if type(criteria) == str:
                print("Comparison not possible")
                return df
            else:
                df = df[df[column_name] < criteria]
                return df
        case ">":
            if type(criteria) == str:
                print("Comparison not possible")
                return df
            else:
                df = df[df[column_name] > criteria]
                return df
        case "<=":
            if type(criteria) == str:
                print("Comparison not possible")
                return df
            else:
                df = df[df[column_name] <= criteria]
                return df
        case ">=":
            if type(criteria) == str:
                print("Comparison not possible")
                return df
            else:
                df = df[df[column_name] >= criteria]
                return df
        case "!=":
            df = df[df[column_name] != criteria]
            return df

        