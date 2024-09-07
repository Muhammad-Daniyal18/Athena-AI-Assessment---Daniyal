import functions
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

#Split into train and test sets
def split_data(df):

    columns_list = list(df.columns)
    y_column = functions.popup_input(f"If you've cleaned up the dataset, please specify the column for your output or y values {columns_list}: ")
    if y_column is None or y_column == "":
        print("Action Cancelled by the user")
        return  
    if y_column not in columns_list:
        while(y_column not in columns_list):
            if y_column is None or y_column == "":
                print("Action cancelled by the user.")
                return df, df  # Exit the loop if "Cancel" is clicked
            y_column = functions.popup_input(f"Column name {y_column} not found in the columns. Please input one of these {columns_list}: ")

    X = df.copy().drop(columns=[y_column])
    y = df.copy()[y_column]

    ratio = functions.popup_input("Please enter your test data size ratio (between 0 and 1, eg. 0.2): ")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(ratio), random_state=42)
    return X_train, X_test, y_train, y_test

#Return Classification evaluation metrics
def evaluation_classification(y_test, y_pred):
    return accuracy_score(y_test,y_pred), f1_score(y_test,y_pred, average='weighted'), confusion_matrix(y_test,y_pred)

#Return Regression evaluation metrics
def evaluation_regression(y_test, y_pred):
    return mean_squared_error(y_test,y_pred), r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred)

#Random Forest Classifier
def RFC(X_train,X_test,y_train,y_test):
    n_estimators = 100
    usual_hyperparameters = ["n_estimators", 100]

    for i in range(0, len(usual_hyperparameters), 2):
        usual_hyperparameters[i+1] = int(functions.popup_input(f"Please enter the hyperparameters for Random Forest Classifier model. \n {usual_hyperparameters[i]} = ? (Usually {usual_hyperparameters[i+1]})"))

    rfc = RandomForestClassifier(n_estimators = usual_hyperparameters[1], random_state=42)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    return evaluation_classification(y_test, y_pred)

#XGBoost Classifier
def XGBC(X_train,X_test,y_train,y_test):

    n_estimators = 100
    learning_rate = 0.3
    usual_hyperparameters = ["n_estimators", 100, "learning_rate", 0.3]
    for i in range(0, len(usual_hyperparameters), 2):
        if i==2:
            usual_hyperparameters[i+1] = float(functions.popup_input(f"Please enter the hyperparameters for XGBoost Classifier model. \n {usual_hyperparameters[i]} = ? (Usually {usual_hyperparameters[i+1]})"))
        else:
            usual_hyperparameters[i+1] = int(functions.popup_input(f"Please enter the hyperparameters for XGBoost Classifier model. \n {usual_hyperparameters[i]} = ? (Usually {usual_hyperparameters[i+1]})"))

    xgb_clf = xgb.XGBClassifier(n_estimators = usual_hyperparameters[1], learning_rate = usual_hyperparameters[3] , random_state=42)
    xgb_clf.fit(X_train, y_train)
    y_pred = xgb_clf.predict(X_test)

    return evaluation_classification(y_test, y_pred)

#Support Vector Machine Classifier
def SVMC(X_train,X_test,y_train,y_test):

    kernel = 'rbf'
    usual_hyperparameters = ["kernel", 'rbf']

    for i in range(0, len(usual_hyperparameters), 2):
        usual_hyperparameters[i+1] = functions.popup_input(f"Please enter the hyperparameters for SVM Classifier model. \n {usual_hyperparameters[i]} = ? (Usually {usual_hyperparameters[i+1]}). Other common options include 'linear', 'poly', 'sigmoid'")

    svm_clf = SVC(kernel=usual_hyperparameters[1], random_state=42)
    svm_clf.fit(X_train, y_train)
    y_pred = svm_clf.predict(X_test)
    
    return evaluation_classification(y_test, y_pred)

#Random Forest Regressor
def RFR(X_train,X_test,y_train,y_test):
    n_estimators = 100
    usual_hyperparameters = ["n_estimators", 100]

    for i in range(0, len(usual_hyperparameters), 2):
        usual_hyperparameters[i+1] = int(functions.popup_input(f"Please enter the hyperparameters for Random Forest Regressor model. \n {usual_hyperparameters[i]} = ? (Usually {usual_hyperparameters[i+1]})"))

    rfr = RandomForestRegressor(n_estimators=usual_hyperparameters[1], random_state=42)
    rfr.fit(X_train, y_train)
    y_pred = rfr.predict(X_test)
    return evaluation_regression(y_test, y_pred)

#XGBoost Regressor
def XGBR(X_train,X_test,y_train,y_test):
    n_estimators = 100
    learning_rate = 0.3
    usual_hyperparameters = ["n_estimators", 100, "learning_rate", 0.3]
    for i in range(0, len(usual_hyperparameters), 2):
        if i==2:
            usual_hyperparameters[i+1] = float(functions.popup_input(f"Please enter the hyperparameters for XGBoost Regressor model. \n {usual_hyperparameters[i]} = ? (Usually {usual_hyperparameters[i+1]})"))
        else:
            usual_hyperparameters[i+1] = int(functions.popup_input(f"Please enter the hyperparameters for XGBoost Regressor model. \n {usual_hyperparameters[i]} = ? (Usually {usual_hyperparameters[i+1]})"))

    xgb_reg = xgb.XGBRegressor(n_estimators = usual_hyperparameters[1], learning_rate = usual_hyperparameters[3], random_state=42)
    xgb_reg.fit(X_train, y_train)
    y_pred = xgb_reg.predict(X_test)
    return evaluation_regression(y_test, y_pred)

#Decision Tree Regressor
def DTR(X_train,X_test,y_train,y_test):
    min_samples_split = 2
    usual_hyperparameters = ["min_samples_split", 2]

    for i in range(0, len(usual_hyperparameters), 2):
        usual_hyperparameters[i+1] = int(functions.popup_input(f"Please enter the hyperparameters for Decision Tree Regressor model. \n {usual_hyperparameters[i]} = ? (Usually {usual_hyperparameters[i+1]})"))

    dtr = DecisionTreeRegressor(min_samples_split= usual_hyperparameters[1] ,random_state=42)
    dtr.fit(X_train, y_train)
    y_pred = dtr.predict(X_test)
    return evaluation_regression(y_test, y_pred)