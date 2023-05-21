import pandas as pd
import spacy
from collections import Counter

# Load the Spanish NLP model from spaCy
nlp = spacy.load("es_core_news_sm")

def extract_keywords(text):
    doc = nlp(text)
    return [token.text for token in doc if token.pos_ in ["NOUN", "PROPN", "VERB"]]

def main():
    # Read the dataset
    data = pd.read_csv("output_data.csv")

    # Extract keywords and phrases
    data["Keywords"] = data["Spanish"].apply(extract_keywords)

    # Calculate the frequency distribution of keywords
    keyword_counter = Counter()
    for keywords in data["Keywords"]:
        keyword_counter.update(keywords)

    # Select the most common keywords
    most_common_keywords = set([kw for kw, _ in keyword_counter.most_common(50)])

    # Filter the dataset to only include sentences with the most common keywords
    data = data[data["Keywords"].apply(lambda x: bool(most_common_keywords.intersection(x)))]

    # Save the preprocessed dataset to a new CSV file
    data.to_csv("preprocessed_data.csv", index=False)

    print("Preprocessing complete. Preprocessed data saved to 'preprocessed_data.csv'.")

if __name__ == "__main__":
    main()