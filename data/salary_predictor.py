# Salary Prediction Based on Years of Experience
# ----------------------------------------------------
# Trains a Linear Regression model and visualizes results.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load dataset
df = pd.read_csv(r"D:\Study\Projects\salary_prediction_project\src\sample_salary_data.csv")
df.columns = ["experience", "salary"]

# Feature and label selection
X = df[["experience"]].values
y = df["salary"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Categorize salaries
bins = [0, 40000, 70000, float("inf")]
labels = ["Low", "Medium", "High"]
y_test_cat = pd.cut(y_test, bins=bins, labels=labels)
y_pred_cat = pd.cut(y_pred, bins=bins, labels=labels)

# Confusion matrix
cm = confusion_matrix(y_test_cat, y_pred_cat, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Categorized Salaries)")
plt.show()

# Regression graph
plt.figure(figsize=(10, 5))
plt.scatter(X_test, y_test, color="blue", label="Actual")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Predicted")
plt.xlabel("Experience (Years)")
plt.ylabel("Salary (Rs.)")
plt.title("Regression: Salary vs Experience")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Predict salary for new entrant
def predict_salary(experience_years):
    salary = model.predict(np.array([[experience_years]]))[0]
    print(f"Predicted salary for {experience_years} years of experience is Rs. {salary:.2f}")

# Example usage
predict_salary(5.2)

if __name__ == "__main__":
    experience_input = float(input("Enter years of experience: "))
    predict_salary(experience_input)
