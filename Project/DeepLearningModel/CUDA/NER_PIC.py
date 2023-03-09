#NER PERSONAL INFORMATION CENSOR MODEL PROTOTYPE

import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForTokenClassification
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import re

# Step 1: Install necessary libraries and dependencies

# You can install the required libraries and dependencies as follows:
# !pip install torch
# !pip install transformers
# !pip install datasets

!wget https://korquad.github.io/dataset/KorEng-CV.tar.gz
!tar -zxvf KorEng-CV.tar.gz

# Step 2: Prepare the dataset

# Load the dataset
dataset = load_dataset("text", data_files={
                        "train": "path_to_train_dataset.txt",
                        "test": "path_to_test_dataset.txt",
                        "validation": "path_to_validation_dataset.txt"})

# Define the labels we want to identify and censor
labels = ["name", "address", "phone", "email", "ssn", "passport", "dl", "biometric", "financial", "health"]

# Tokenize the dataset
def tokenize_dataset(examples):
    return tokenizer(examples["text"], truncation=True, is_split_into_words=True)

tokenized_dataset = dataset.map(tokenize_dataset, batched=True, num_proc=4)

# Define the function to align labels with tokenized data
def align_labels_with_tokenization(example):
    word_ids = example.word_ids()
    sequence = example.sequence
    labels = example["label"]

    # create a list of labels for each subword token
    subword_labels = [-100] * len(sequence)

    for word_idx, (start, end) in enumerate(word_ids):
        # the end index is exclusive in the word_ids
        # the last token of each word should have the label of the word
        if end is not None:
            subword_labels[start:end] = [labels[word_idx]] * (end - start)

    return {"input_ids": sequence, "labels": subword_labels}

tokenized_dataset = tokenized_dataset.map(
    align_labels_with_tokenization, batched=True, num_proc=4, remove_columns=["text"])

# Set the data type to PyTorch tensors
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Step 3: Fine-tune the pre-trained NER model

# Load the pre-trained model and tokenizer
model = XLMRobertaForTokenClassification.from_pretrained("xlm-roberta-base", num_labels=len(labels))
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=5,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
)

# Define the function to encode the text data
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["text"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["label"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if "X" in labels[-1] else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Encode the dataset
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Split the dataset
train_dataset = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["validation"]

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                               'attention_mask': torch.stack([f[1] for f in data]),
                               'labels': torch.stack([f[2] for f in data])},
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

import os
import json
import docx2txt
import PyPDF2
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForTokenClassification
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Step 4: Use the fine-tuned model to censor text

# Load the pre-trained model and tokenizer
model = XLMRobertaForTokenClassification.from_pretrained("xlm-roberta-base", num_labels=len(labels))
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

# Define a function to preprocess text before tokenization
def preprocess_text(text):
    # Remove newlines
    text = text.replace('\n', ' ')
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def censor_text(input_file_path, output_file_path):
    # Check file extension
    ext = os.path.splitext(input_file_path)[1].lower()
    if ext not in ['.txt', '.docx', '.pdf']:
        print("Unsupported file extension:", ext)
        return

    # Read file contents
    if ext == '.txt':
        with open(input_file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    elif ext == '.docx':
        text = docx2txt.process(input_file_path)
    else:
        with open(input_file_path, 'rb') as f:
            reader = PyPDF2.PdfFileReader(f)
            text = ''
            for page in reader.pages:
                text += page.extract_text()

    # Preprocess text before tokenization
    text = preprocess_text(text)

    # Tokenize the text
    tokens = tokenizer.encode(text, add_special_tokens=False)

    # Make predictions using the fine-tuned model
    inputs = torch.tensor([tokens])
    with torch.no_grad():
        predictions = model(inputs)[0]

    # Convert predictions to labels
    predicted_labels = predictions.argmax(dim=2)[0]

    # Create a list of words and their labels
    words = tokenizer.convert_ids_to_tokens(tokens)
    labels = [labels[predicted_label] for predicted_label in predicted_labels]

    # Censor personal information
    censored_words = []
    for word, label in zip(words, labels):
        if label in ["name", "address", "phone", "email", "ssn", "passport", "dl", "biometric", "financial", "health"]:
            censored_words.append("[CENSORED]")
        else:
            censored_words.append(word)

    # Join censored words into text
    censored_text = " ".join(censored_words)

    # Write censored text to file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(censored_text)

    print("Censored text saved to:", output_file_path)
