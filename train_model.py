import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data.csv")  # הקובץ המקורי שלך
X = df.drop("Class", axis=1)
y = df["Class"]

model = LogisticRegression(max_iter=200)
model.fit(X, y)

joblib.dump(model, "model.pkl")  # 🔒 שומר לקובץ

