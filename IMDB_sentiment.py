import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox

# ------------------------
# 1. Load Dataset
# ------------------------
df = pd.read_csv("IMDB Dataset.csv")

# ------------------------
# 2. Preprocessing
# ------------------------
def clean_text(text):
    text = re.sub(r"<.*?>", " ", text)  # remove HTML
    text = text.lower()  # lowercase
    return text

df["review"] = df["review"].apply(clean_text)
X = df["review"]
y = df["sentiment"].map({"positive": 1, "negative": 0})

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------
# 3. Vectorization
# ------------------------
vectorizer = CountVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ------------------------
# 4. Train Models
# ------------------------
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_vec, y_train)
y_pred_lr = log_reg.predict(X_test_vec)
acc_lr = accuracy_score(y_test, y_pred_lr)

nb = MultinomialNB()
nb.fit(X_train_vec, y_train)
y_pred_nb = nb.predict(X_test_vec)
acc_nb = accuracy_score(y_test, y_pred_nb)

print("=== Model Accuracies ===")
print(f"Logistic Regression: {acc_lr:.4f}")
print(f"Naive Bayes:        {acc_nb:.4f}")

# ------------------------
# 5. Tkinter UI
# ------------------------
class SentimentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("IMDb Sentiment Classifier")
        self.root.geometry("700x500")
        self.root.configure(bg="#f0f0f0")

        tk.Label(root, text="Enter Movie Review:", font=("Helvetica", 14), bg="#f0f0f0").pack(pady=10)
        self.text_input = tk.Text(root, height=8, width=70, font=("Helvetica", 12))
        self.text_input.pack(pady=10)

        tk.Button(root, text="Predict Sentiment", font=("Helvetica", 14, "bold"),
                  bg="#4CAF50", fg="white", command=self.predict).pack(pady=10)

        self.output_label = tk.Label(root, text="", font=("Helvetica", 16, "bold"), bg="#f0f0f0")
        self.output_label.pack(pady=20)

        tk.Label(root, text=f"Model Accuracies (LR: {acc_lr:.2f}, NB: {acc_nb:.2f})",
                 font=("Helvetica", 10), bg="#f0f0f0").pack(side="bottom", pady=5)

    def predict(self):
        review = self.text_input.get("1.0", tk.END).strip()
        if not review:
            messagebox.showwarning("Input Error", "Please enter a review.")
            return

        review_vec = vectorizer.transform([clean_text(review)])
        prediction = log_reg.predict(review_vec)[0]
        proba = log_reg.predict_proba(review_vec)[0]
        confidence = max(proba) * 100

        if prediction == 1:
            self.output_label.config(text=f"Positive Review ✅\nConfidence: {confidence:.2f}%", fg="green")
        else:
            self.output_label.config(text=f"Negative Review ❌\nConfidence: {confidence:.2f}%", fg="red")

if __name__ == "__main__":
    root = tk.Tk()
    app = SentimentApp(root)
    root.mainloop()
