import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from transformers import Trainer, TrainingArguments

import torch

class SpanishEnglishDataset(Dataset):
    def __init__(self, data, tokenizer, source_lang="Spanish", target_lang="English"):
        self.data = data
        self.tokenizer = tokenizer
        self.source_lang = source_lang
        self.target_lang = target_lang

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source_text = self.data.loc[idx, self.source_lang]
        target_text = self.data.loc[idx, self.target_lang]

        source_tokenized = self.tokenizer.encode_plus(
            source_text, return_tensors="pt", padding="max_length", truncation=True
        )

        target_tokenized = self.tokenizer.encode_plus(
            target_text, return_tensors="pt", padding="max_length", truncation=True
        )

        return {
            "input_ids": source_tokenized["input_ids"].flatten(),
            "attention_mask": source_tokenized["attention_mask"].flatten(),
            "labels": target_tokenized["input_ids"].flatten(),
        }

def main():
    data_file = "output_data.csv"
    data = pd.read_csv(data_file)

    # Initialize tokenizer and model
    model_name = "Helsinki-NLP/opus-mt-es-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Create dataset
    dataset = SpanishEnglishDataset(data, tokenizer)

    # Training settings
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=6,
        per_device_train_batch_size=16,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Train the model
    trainer.train()

if __name__ == "__main__":
    main()