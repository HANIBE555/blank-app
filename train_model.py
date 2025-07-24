import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
st.markdown(
    """
    <style>
    .main {
        direction: rtl;
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# קריאת הקובץ
df = pd.read_csv("all_smote_resampled_data.csv")
X = df.drop("Class", axis=1)
y = df["Class"]

# הגדרת K-Fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# הגדרת המודל
model = LogisticRegression(max_iter=200)

# ביצוע Cross Validation – מודל יאומן מחדש בכל קיפול, ונשמור את האחרון
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train, y_train)

# שמירה של המודל המאומן האחרון
joblib.dump(model, "model.pkl")
