import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# Define your custom dataset class
class ToxicDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {key: torch.tensor(val[i]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[i])
        return item

# Load the data
data = pd.read_csv('train.csv')
train_data, validation_data = train_test_split(data, test_size=0.2, random_state=42)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def encode_inputs(batch):
    comments = batch['comment_text'].tolist()
    return tokenizer(comments, padding=True, truncation=True, max_length=512)

train_encoded_data = encode_inputs(train_data)
validation_encoded_data = encode_inputs(validation_data)

train_dataset = ToxicDataset(train_encoded_data, train_data.iloc[:, 2:].values)
validate_dataset = ToxicDataset(validation_encoded_data, validation_data.iloc[:, 2:].values)

# Load the pre-trained model
model_name = 'distilbert-base-uncased'
config = DistilBertConfig.from_pretrained(model_name, num_labels=6)
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

# Training arguments
train_arguments = TrainingArguments(
    evaluation_strategy="steps",
    output_dir="output",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir='logs',
    logging_steps=100
)

# Trainer
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=validate_dataset,
    args=train_arguments
)

trainer.train()

save_directory = 'saved'
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)
tokenizer = AutoTokenizer.from_pretrained(save_directory)
model = AutoModelForSequenceClassification.from_pretrained(save_directory)

results = trainer.evaluate()
print(results)
