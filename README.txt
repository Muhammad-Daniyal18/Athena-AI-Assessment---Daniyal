Problem understanding:
The problem in hand is to develop an easy solution for users without a machine learning background to allow them to train models on their own, without having to write a single bit of code. All they are needed to do is to provide the dataset, do some assisted filtering, choose how they want to split the dataset, what they want the hyperparameters to be, and train on a regression or classification model of their choice. As results are visible to them, they should be able to save whichever one, or multiple they like. 

Architectural design:
My architecture is such that I formed multiple user interface windows or frames using the tkinter library. I laid down all the functionalities the user would need and formed buttons and the menu-bar according to that. To add functionalities to the buttons, I coded their respective functions in the functions.py file. For the model-training process, I kept all the models in the models.py file where I trained on the dataset provided and returned the results. A results.json file is populated and updated as soon as the user saves any model results.
For every functionality, there’s a button present in the interface and the processes are intuitive enough for the users to undergo. 

Technical decisions:
Using tkinter, we can make user-friendly interfaces, popup windows, textboxes etc. It also gives a lot of freedom to develop the aesthetics of the interface, hence I made use of that library.
Used pandas because of the plethora of options it provides to handle dataframes. 
As asked, I implemented 3 classification models: Random Forest, XGBoost and SVM and 3 regression models: Random Forest, XGBoost and Decision Tree. My decision to use these models was taken because of the wide range of options these models provide. For any kind of user input, at least one of these models will be best-suited.
Error handling at inputs and other avenues is done to a decent extent. Wherever important, the user will be asked again and again to enter the correct values. At other points, a very clear message on what is expected Is displayed, and the user is expected to follow them to not get into errors. The program doesn’t crash however, at most, the undergoing function will be terminated for wrong and unhandled inputs.

Exceeding the requirements:
I felt like there is no process of modification which does not come with the possibility of mistakenly changing something you didn’t want to. Or having the strong urge to turn back time and change your steps taken. That is why I implemented the Undo and Redo functions with the capability of reversing the last changing action. If an action did not have an effect, the undo functionality will not count that as an action, and still undo the previous action that brought about a change.
At multiple other points, the user is asked to choose for themselves, apart from the specified requirements given.

User Guide
File to run: main.py
Necessary files to include in the directory: models.py, functions.py

All the functionalities required have been implemented, and this user guide will go one-by-one to show you how to test them all.
-	Requirement 1a: Upload a Dataset:
As the main file is run, a window appears with the home screen on (green background). A “Load CSV File” button is present in the middle of the frame, which will help loading a csv file from anywhere in your system.
-	Requirement 1b: Configure Model:
Click on the “Preprocessing” tab in the menu bar, and it will lead you to the dataset modification and preprocessing frame (blue background). After the dataset is loaded, you can:
1.	Drop Columns by clicking on the “Drop Columns” button and providing the column name(s) (multiple allowed).
2.	Specify Ordinal Columns by clicking on the respective button, providing column name(s), and entering the column items in the order of your choice.

-	Requirement 2a: Handle Missing Data:
The third button in the preprocessing frame, as labelled, will help you handle the missing data. It gives you three choices, you can:
1.	Drop all the columns containing the null values
2.	Drop all the rows containing the null values
3.	Enter a custom value to fill in for all null values

-	Requirement 2b: Split Dataset:
Clicking on the “Train” tab from the menu-bar will lead you to a train frame which contains “Split Data” as the second button. Once you click that, first specify what your output column is, or in other words what are you trying to predict with the model. After entering the column name, you are given full choice of your test-data to total-data size ratio. Enter a number ranging (0,1) and your dataset will be split.
Additionally, I have added two viewing buttons to view and confirm your test and train datasets. Click on them to view the respective datasets. Training and testing will be occurring on these datasets.
-	Requirement 3: Data Slicing and Condition Setting:
Now, before sending your dataset to test-train splits and the models, you may want to filter some of the information. That is what the “Filter Data” button on the train frame is there for. How to use?
1.	Enter the column name you want to filter by (eg. Runs)
2.	Choose one of the comparison operators (eg. “>”)
3.	Enter the criteria (eg. 6000)
The example case will give us dataset containing all the rows where Runs > 3000

-	Requirement 4a: Model type selection:
Click on Select Model in the train frame, and the first choice you’d be given would be to choose between Classification and Regression models, as asked.
-	Requirement 4b: Choose Models:
Once you select the model type, the next popup appearing will ask you to choose from 3 models. Type in the number to start training your dataset on it.
-	Requirement 4c: Train Model:
After an intermediary step of specifying your model hyperparameter(s), the model starts training and a popup window appears with the relevant results (accuracy score, f1 score, confusion matrix, r2 score, mean squared error, mean absolute error).
-	Requirement 5a: Hyperparameter Tuning:
Once you select your model, you will be asked to set its most important hyperparameter(s) in the next popup. If you don’t have an idea of what to enter, the default value is also displayed to aid you.
-	Requirement 5b: Model Evaluation:
As the model is trained, results are displayed with a “Save Results” button in the same window. If you like the scores, you can save the results for that model by clicking on the button. The results.json file will be populated. You can run different models and their results will be saved separately for you to compare as well. However, if you save results for the same model again, they will overwrite the previous results of that model in the results file. 
-	Requirement 6: User Interface:
As allowed, I used tkinter library for the user interface.
-	Requirement 7: Libraries:
As allowed, I only used tkinter, numpy, pandas and scikit learn libraries.



