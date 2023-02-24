# Define the task you want to perform, which is to identify personal information such as names, phone numbers, e-mails, 
# resident registration numbers, and credit card numbers from text data.

task = "censor_personal_info"

# Choose a model or analysis method that is appropriate for your problem and data. For example, you might choose a deep 
# learning model like a convolutional neural network (CNN) or a recurrent neural network (RNN) with attention mechanisms.

import subprocess

def ensure_packages_installed(packages):
    """
    Checks if the specified packages are installed and up-to-date. If not, installs or upgrades them.
    """
    for package in packages:
        try:
            # Check if the package is installed and get its version
            result = subprocess.run(["pip", "show", package], capture_output=True, check=True, text=True)
            installed_version = None
            for line in result.stdout.splitlines():
                if line.startswith("Version:"):
                    installed_version = line.split(":")[1].strip()
                    break

            # Check if the package is up-to-date
            latest_version = subprocess.check_output(["pip", "install", "--no-cache-dir", "--upgrade", package, "-q", "--no-input"], stderr=subprocess.STDOUT).decode().strip()
            if installed_version is not None and latest_version == installed_version:
                print(f"{package} is already up-to-date ({latest_version}).")
            else:
                print(f"{package} was updated from {installed_version} to {latest_version}.")
        except subprocess.CalledProcessError as e:
            # Package is not installed, so install it
            print(f"{package} is not installed, so installing...")
            subprocess.run(["pip", "install", "--no-cache-dir", package, "-q", "--no-input"], check=True)

    print("All packages are installed and up-to-date!")
    
packages = ["numpy", "pandas", "matplotlib"] #사용법 예시 다른 패키지로 변경가능
ensure_packages_installed(packages)


from transformers import pipeline

nlp_model = "distilbert-base-uncased"

ner_model = pipeline("ner", model=nlp_model, device=0)

# Collect a dataset of text data that contains personal information, and preprocess the data by tokenizing and normalizing the text.

import pandas as pd

# Load data from a CSV file
data = pd.read_csv('data.csv')

# Preprocess the text data
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(nlp_model)

# Tokenize the text data
tokenized_data = [tokenizer(text) for text in data['text']]

# Convert the tokenized data to tensors
import torch

input_ids = torch.tensor([text['input_ids'] for text in tokenized_data])
attention_masks = torch.tensor([text['attention_mask'] for text in tokenized_data])

# Start with a small set of labeled data that contains personal information. This can be done by manually annotating a subset 
# of the data, or by using an existing labeled dataset.

# Manually annotate a subset of the data
labeled_data = [{'text': 'John Smith is a customer of our company.', 'labels': [(0, 10, 'PERSON')]},
                {'text': 'Call us at 555-1234 to speak with a representative.', 'labels': [(8, 18, 'PHONE_NUMBER')]},
                {'text': 'Email us at john.smith@example.com for more information.', 'labels': [(11, 25, 'EMAIL')]}]

# Implement an active learning loop that selects a subset of the unlabeled data to be annotated by a human annotator. 
# The NLP model is then retrained on the expanded labeled dataset, and the loop continues until the desired level 
# of performance is reached.

import random

# Define a function to select a subset of unlabeled data for annotation
def select_data_for_annotation(unlabeled_data, n):
    return random.sample(unlabeled_data, n)

# Define a function to retrain the NLP model on the expanded labeled dataset
def retrain_nlp_model(labeled_data):
    # Convert the labeled data to tensors
    labels = []
    for example in labeled_data:
        text = example['text']
        entities = example['labels']
        entity_labels = [0] * len(text)
        for entity in entities:
            start, end, label = entity
            entity_labels[start:end] = [1] * (end - start)
        labels.append(entity_labels)

    labels = torch.tensor(labels)

    # Retrain the NLP model on the expanded labeled dataset
    from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

    model = AutoModelForTokenClassification.from_pretrained(nlp_model, num_labels=2)
    training_args = TrainingArguments(
        output_dir='./results',          
        num_train_epochs=1,              
        per_device_train_batch_size=16,  
        per_device_eval_batch_size=64,   
        warmup_steps=500,                
        weight_decay=0.01,               
        logging_dir='./logs',            
        logging_steps=10,
    )
    trainer = Trainer(
        model=model,                         
        args=training_args,                  
        train_dataset=(input_ids, attention_masks, labels),
        eval_dataset=(input_ids, attention_masks, labels),
        compute_metrics=lambda p: {'accuracy': (p.predictions.argmax(-1) == p.label_ids).float().mean()}
    )
    trainer.train()
    return model

# Once the NLP model has been trained to identify personal information entities, use it to censor the personal information 
# in new text data by replacing the entities with a mask token.

# Define a function to censor personal information in text data
def censor_personal_info(model, tokenizer, text, mask_token="[MASK]"):
    # Tokenize the text
    tokenized_text = tokenizer(text, return_offsets_mapping=True, padding=True, truncation=True)

    # Get the model's predictions
    input_ids = torch.tensor([tokenized_text['input_ids']])
    attention_mask = torch.tensor([tokenized_text['attention_mask']])
    predictions = model(input_ids, attention_mask)

    # Replace the entities with a mask token
    token_offsets = tokenized_text['offset_mapping']
    censored_text = ""
    for i in range(len(predictions[0])):
        token = tokenizer.decode(input_ids[0][i])
        entity_label = predictions[0][i].argmax().item()
        start_offset, end_offset = token_offsets[i]
        if entity_label == 1:
            censored_text += mask_token * (end_offset - start_offset)
        else:
            censored_text += token[start_offset:end_offset]

    return censored_text

# To evaluate the performance of the NLP model, we can use metrics such as precision, recall, and F1 score. 
# We'll also need a labeled dataset with ground truth labels for personal information entities.

from sklearn.metrics import classification_report

# Load the test data with ground truth labels for personal information entities
test_data = load_data(test_file)

# Tokenize the test data
tokenized_test = [tokenizer(data["text"], return_offsets_mapping=True) for data in test_data]

# Convert the ground truth labels to tensors
labels = [torch.tensor(data["labels"]) for data in test_data]

# Evaluate the model's predictions on the test data
model.eval()
with torch.no_grad():
    predictions = [model(torch.tensor([text["input_ids"]]), torch.tensor([text["attention_mask"]])).squeeze(0).argmax(axis=1) for text in tokenized_test]

# Flatten the predictions and labels
flat_predictions = [p.item() for sublist in predictions for p in sublist]
flat_labels = [l.item() for sublist in labels for l in sublist]

# Print the classification report
print(classification_report(flat_labels, flat_predictions, target_names=["not_personal_info", "personal_info"]))

# If the performance of the NLP model is not satisfactory, we can fine-tune it on additional labeled data. 
# We'll need to load the additional data and update the model's parameters using a fine-tuning procedure.

from transformers import AdamW
from torch.utils.data import DataLoader

# Load the additional labeled data
additional_data = load_data(additional_file)

# Tokenize the additional data
tokenized_additional = [tokenizer(data["text"], return_offsets_mapping=True) for data in additional_data]
additional_labels = [torch.tensor(data["labels"]) for data in additional_data]

# Create a PyTorch dataset and dataloader for the additional data
additional_dataset = NERDataset(tokenized_additional, additional_labels)
additional_dataloader = DataLoader(additional_dataset, batch_size=batch_size, shuffle=True)

# Fine-tune the model on the additional data
model.train()
optimizer = AdamW(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    for batch in additional_dataloader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        optimizer.zero_grad()
        loss = model(input_ids, attention_mask, labels=labels).loss
        loss.backward()
        optimizer.step()

# Once the NLP model has been trained and fine-tuned (if necessary), we can save it to disk for future use.

import os

# Create a directory to save the model
os.makedirs(model_dir, exist_ok=True)

# Save the tokenizer and model to disk
tokenizer.save_pretrained(model_dir)
model.save_pretrained(model_dir)

# To load the NLP model and tokenizer for future use, we can use the from_pretrained() method.

from transformers import AutoTokenizer, AutoModelForTokenClassification

# Load the tokenizer and model from disk
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForTokenClassification.from_pretrained(model_dir)

# Test the loaded model
model.eval()
with torch.no_grad():
    input_ids = tokenizer.encode("John Smith is a data scientist", return_tensors="pt")
    output = model(input_ids).logits
    predictions = output.argmax(dim=2)
    print(tokenizer.convert_ids_to_tokens(input_ids[0]))
    print(predictions[0])

# To use the trained NLP model to censor personal information in new text, we can use the predict() function we defined earlier.

def censor_personal_info(text):
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt")
    # Make predictions using the model
    outputs = model(**inputs)
    # Get the predicted labels
    predicted_labels = torch.argmax(outputs.logits, dim=2).squeeze()
    # Create a mask to censor the predicted entities
    mask = predicted_labels == label_map["PER"] | predicted_labels == label_map["PHONE"] | predicted_labels == label_map["EMAIL"] | predicted_labels == label_map["SSN"] | predicted_labels == label_map["CREDIT_CARD"]
    # Censor the entities in the text
    censored_text = ""
    for token, m in zip(inputs["input_ids"][0], mask):
        if m:
            censored_text += "[REDACTED]"
        else:
            censored_text += tokenizer.decode([token])
    return censored_text


# To test the `censor_personal_info()` function, we can call it with some example text.

text = "My name is John Smith and my phone number is 555-1234. My email address is john.smith@example.com and my credit card number is 1234-5678-9012-3456."
censored_text = censor_personal_info(text)
print(censored_text)


# If the initial training dataset is not large enough to provide satisfactory performance, we can fine-tune the model using additional annotated data.

# Load the fine-tuning dataset
dataset = load_dataset("csv", data_files={"train": "fine_tune_data.csv"})

# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Fine-tune the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
trainer.train()


# Once the model has been fine-tuned (if necessary), we can evaluate its performance on a separate test dataset.

# Load the test dataset
test_dataset = load_dataset("csv", data_files={"test": "test_data.csv"})

# Tokenize the test dataset
tokenized_test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)

# Evaluate the model
eval_results = trainer.evaluate(tokenized_test_dataset["test"])
print(eval_results)


# Once the NLP model has been trained and evaluated, we can deploy it for use in a production environment.

# Load the production dataset
production_dataset = load_dataset("csv", data_files={"production": "production_data.csv"})

# Tokenize the production dataset
tokenized_production_dataset = production_dataset.map(tokenize_and_align_labels, batched=True)

# Use the model to make predictions on the production dataset
predictions = trainer.predict(tokenized_production_dataset["production"])

# Save the predictions to disk
with open("predictions.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "text", "labels"])
    for example_id, example_text, example_labels in zip(predictions["id"], predictions["text"], predictions["label"]):
        writer.writerow([example_id, example_text, example_labels])

# Load the predictions from disk
with open("predictions.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip the header row
    predictions = []
    for row in reader:
        predictions.append({
            "id": int(row[0]),
            "text": row[1],
            "labels": row[2],
        })

# Post-process the predictions
for prediction in predictions:
    prediction["censored_text"] = censor_personal_info(prediction["text"], prediction["labels"])

# Save the censored predictions to disk
with open("censored_predictions.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "text", "labels", "censored_text"])
    for prediction in predictions:
        writer.writerow([prediction["id"], prediction["text"], prediction["labels"], prediction["censored_text"]])



