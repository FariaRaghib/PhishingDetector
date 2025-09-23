# ML_models.py

# Step 1 import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import svm, tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


# ====== Define models ====== #
nb_model = GaussianNB()
svm_model = SVC(probability=True)
dt_model = DecisionTreeClassifier()
rf_model = RandomForestClassifier()
ab_model = AdaBoostClassifier()
nn_model = MLPClassifier(max_iter=500)
kn_model = KNeighborsClassifier()

df = None
phishing_df = None
legitimate_df = None
def calculate_measures(TN, TP, FN, FP):
    """Helper to calculate metrics safely."""
    model_accuracy = (TP + TN) / (TP + TN + FN + FP)
    model_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    model_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    return model_accuracy, model_precision, model_recall


def load_data():
    """Load and preprocess data."""
    global df,phishing_df, legitimate_df
    legitimate_df = pd.read_csv("structured_data_legitimate.csv")
    phishing_df = pd.read_csv("structured_data_phishing.csv")

    df = pd.concat([legitimate_df, phishing_df], axis=0)
    df = df.sample(frac=1)
    df = df.drop('URL', axis=1)
    df = df.drop_duplicates()
    df = df.fillna(0)

    X = df.drop('label', axis=1)
    Y = df['label']
    return X, Y


def train_and_evaluate():
    """Train models and print evaluation results with K-fold CV."""
    X, Y = load_data()

    # Split data for initial test
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=10
    )

    # Example: Train a Linear SVM
    svm_model = svm.LinearSVC(max_iter=5000)
    svm_model.fit(x_train, y_train)
    predictions = svm_model.predict(x_test)

    cm = confusion_matrix(y_true=y_test, y_pred=predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc, prec, rec = calculate_measures(tn, tp, fn, fp)

    print("Initial Linear SVM")
    print("accuracy -->", acc)
    print("precision -->", prec)
    print("recall -->", rec)
    print("-" * 50)

    # --- K-fold cross validation ---
    K = 5
    total = X.shape[0]
    index = int(total / K)

    X_train_list = [
        X.iloc[np.r_[0:i*index, (i+1)*index:]] for i in range(K)
    ]
    X_test_list = [
        X.iloc[i*index:(i+1)*index] for i in range(K)
    ]
    Y_train_list = [
        Y.iloc[np.r_[0:i*index, (i+1)*index:]] for i in range(K)
    ]
    Y_test_list = [
        Y.iloc[i*index:(i+1)*index] for i in range(K)
    ]

    # Storage for results
    results = {
        "RF": (rf_model, [], [], []),
        "DT": (dt_model, [], [], []),
        "SVM": (svm.SVC(), [], [], []),
        "AB": (ab_model, [], [], []),
        "NB": (nb_model, [], [], []),
        "NN": (nn_model, [], [], []),
        "KN": (kn_model, [], [], []),
    }

    # Run CV
    for i in range(K):
        for key, (model, acc_list, prec_list, rec_list) in results.items():
            model.fit(X_train_list[i], Y_train_list[i])
            preds = model.predict(X_test_list[i])
            cm = confusion_matrix(
                y_true=Y_test_list[i], y_pred=preds, labels=[0, 1]
            )
            tn, fp, fn, tp = cm.ravel()
            acc, prec, rec = calculate_measures(tn, tp, fn, fp)
            acc_list.append(acc)
            prec_list.append(prec)
            rec_list.append(rec)

    # Summarize
    for key, (model, acc_list, prec_list, rec_list) in results.items():
        print(f"{key} accuracy ==> ", sum(acc_list) / K)
        print(f"{key} precision ==> ", sum(prec_list) / K)
        print(f"{key} recall ==> ", sum(rec_list) / K)
        print("-" * 50)

    # Plot
    data = {
        "accuracy": [sum(results[k][1]) / K for k in results],
        "precision": [sum(results[k][2]) / K for k in results],
        "recall": [sum(results[k][3]) / K for k in results],
    }
    index_labels = list(results.keys())

    df_results = pd.DataFrame(data=data, index=index_labels)
    ax = df_results.plot.bar(rot=0)
    plt.show()
X, Y = load_data()
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

# ========= Train Models ========= #
nb_model.fit(x_train, y_train)
svm_model.fit(x_train, y_train)
dt_model.fit(x_train, y_train)
rf_model.fit(x_train, y_train)
ab_model.fit(x_train, y_train)
nn_model.fit(x_train, y_train)
kn_model.fit(x_train, y_train)

# ========= Entry Point ========= #
if __name__ == "__main__":
    train_and_evaluate()
