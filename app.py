import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

class NaiveBayesClassifier:
    def __init__(self):
        self.class_word_probs = {}
        self.class_priors = {}
        self.classes = []

    def preprocess_text(self, text):
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word.lower() not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return tokens

    def train(self, X_train, y_train):
        self.classes = np.unique(y_train)
        total_samples = len(y_train)
        for class_ in self.classes:
            class_samples = X_train[y_train == class_]
            self.class_priors[class_] = len(class_samples) / total_samples
            all_words = [
                word for text in class_samples for word in self.preprocess_text(text)
            ]
            word_counts = pd.Series(all_words).value_counts()
            self.class_word_probs[class_] = (word_counts + 1) / (
                len(all_words) + len(word_counts)
            )

    def predict(self, X_test):
        predictions = []
        for text in X_test:
            text_words = self.preprocess_text(text)
            probs = {
                class_: np.log(self.class_priors[class_])
                + sum(
                    np.log(self.class_word_probs[class_].get(word, 1e-10))
                    for word in text_words
                )
                for class_ in self.classes
            }
            predicted_class = max(probs, key=probs.get)
            predictions.append(predicted_class)
        return predictions

def main():
    st.title("Twitter Sentiment Analysis")

    # Load data
    try:
        data = pd.read_csv("text.csv")  # Adjust file path if necessary
    except FileNotFoundError:
        st.error("Error: File not found. Please ensure the file path is correct.")
        return
    except pd.errors.EmptyDataError:
        st.error("Error: The file is empty.")
        return
    except pd.errors.ParserError:
        st.error("Error: Unable to parse the CSV file.")
        return

    # Check if 'text' column exists
    if "text" not in data.columns:
        st.error("Error: 'text' column not found in the dataset.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"], data["label"], test_size=0.15, random_state=42
    )

    clf = NaiveBayesClassifier()
    clf.train(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = np.mean(y_pred == y_test)

    st.write("## Model Evaluation")
    st.write(f"Accuracy: {accuracy:.2f}")

    keyword = st.text_input("Enter the keyword to search:")

    if keyword:
        search_results = data[data["text"].str.contains(keyword, case=False)]
        st.write("### Search Results")
        if not search_results.empty:
            for index, row in search_results.iterrows():
                st.write(f"**Tweet:** {row['text']}")
                st.write(f"**Label:** {row['label']}")
                st.write("---")
        else:
            st.write("No results found for the given keyword.")

    st.write("## Confusion Matrix")
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Neutral", "Positive"],
        yticklabels=["Negative", "Neutral", "Positive"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    st.pyplot(plt)

def highlight_keyword(text, keyword):
    # Function to highlight keyword in text
    highlighted_text = text.replace(keyword, f"**{keyword}**")
    return highlighted_text

if __name__ == "__main__":
    main()
