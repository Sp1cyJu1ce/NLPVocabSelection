def generate_sentence(text, model, tokenizer, data, unique_keywords, max_len, n_words=10):
    predicted_topic = predict_topic(text, model, tokenizer, max_len)
    predicted_keyword = unique_keywords[predicted_topic]
    filtered_data = data[data["Keywords"] == predicted_keyword]

    if filtered_data.empty:
        print(f"No data available for the predicted topic: {predicted_topic}")
        return []

    sampled_words = filtered_data.sample(min(n_words, len(filtered_data)))["Spanish"].tolist()
    return sampled_words

def main():

    data = pd.read_csv("preprocessed_data.csv")

    model_name = "dccuchile/bert-base-spanish-wwm-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained("topic_classification_model")

    input_text = "Divórciame"
    max_len = 128
    unique_keywords = data["Keywords"].unique()
    vocabulary_list = generate_sentence(input_text, model, tokenizer, data, unique_keywords, max_len)

    print("Generated sentence list for the given input text:")
    print(vocabulary_list)

if __name__ == "__main__":
    main()