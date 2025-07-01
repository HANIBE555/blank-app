import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

def main():
    # קריאת הנתונים מקובץ שהעלית
    df = pd.read_csv("final_data_for_project.csv")
    X = df.drop("Class", axis=1)
    y = df["Class"]

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for i, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # החלת SMOTE על סט האימון בלבד
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X_train_res, y_train_res)

        y_pred = model.predict(X_test)

        if i == 5:  # מציג את הפלט של הקיפול האחרון בלבד
            print(f"--- Evaluation on Fold {i} ---")
            print("Confusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
