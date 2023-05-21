def predict_topic(text, model, tokenizer, max_len):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    
    probabilities = torch.softmax(logits, dim=1).cpu().numpy().flatten()
    predicted_topic = probabilities.argmax()
    
    return predicted_topic

def generate_vocabulary(text, model, tokenizer, keywords_list, max_len):
    predicted_topic = predict_topic(text, model, tokenizer, max_len)
    return keywords_list[predicted_topic]

def main():

    data = pd.read_csv("preprocessed_data.csv")
    model_name = "dccuchile/bert-base-spanish-wwm-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained("topic_classification_model")

    input_text = "Quiero aprender a hablar con chicas"
    max_len = 128
    unique_keywords = data["Keywords"].unique()
    vocabulary_list = generate_vocabulary(input_text, model, tokenizer, unique_keywords, max_len)

    print("Generated vocabulary list for the given input text:")
    print(vocabulary_list)

if __name__ == "__main__":
    main()