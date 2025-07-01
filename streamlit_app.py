import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

# קריאת הקובץ עם הדגימות (נוצר קודם)
df_train = pd.read_csv("train_with_smote.csv")
X_train = df_train.drop("Class", axis=1)
y_train = df_train["Class"]

# קורא את כל הנתונים המקוריים לבדיקה
df = pd.read_csv("final_data_for_project.csv")
X = df.drop("Class", axis=1)
y = df["Class"]

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = LogisticRegression(max_iter=200)

for i, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
    # בשביל להעריך – לא עושים SMOTE כאן, משתמשים בקובץ האימון שנוצר
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    # מאמנים על כל הסט שאספת ב-SMOTE
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if i == 5:  # מציג את הפלט של הקיפול האחרון בלבד
        print(f"\nLogistic Regression + SMOTE (train file) - Evaluation on Fold {i}")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
