!pip install transformers # install the transformers package

from transformers import pipeline

# Load the XLM-RoBERTa NER pipeline with English and Korean language support
ner_pipeline = pipeline("ner", model="xlm-roberta-large-finetuned-conll02-spanish", tokenizer="xlm-roberta-large-finetuned-conll02-spanish", grouped_entities=True)

# Example sentence in Korean and English
text = "저는 David Kim 입니다. My phone number is 555-1234, and my account number is 123456789."

# Use the NER pipeline to extract named entities
named_entities = ner_pipeline(text)

# Print the named entities
for entity in named_entities:
    print(entity)

import random

# Example pool of unlabeled data
unlabeled_data = [
    "안녕하세요. 제 이름은 김영호입니다. 제 전화번호는 010-1234-5678이고, 제 계좌번호는 123-456-789입니다.",
    "Hello, my name is John Smith. You can reach me at 555-1234, and my account number is 987654321.",
    "이메일 주소는 abc123@gmail.com이며, 비밀번호는 1234입니다.",
    "My email is jdoe@example.com and my credit card number is 1234-5678-9012-3456.",
    "저는 김영수입니다. 제 핸드폰 번호는 010-9876-5432이고, 제 주소는 서울시 강남구입니다.",
    "My name is Jane Doe and my social security number is 123-45-6789.",
    "오늘 날씨가 매우 좋네요. 내일은 비가 올 것 같습니다.",
    "This is an example of a tweet that may contain personal information such as a phone number or email address."
]

# Define a function to extract informative samples from the unlabeled data
def select_informative_samples(data, num_samples=5):
    informative_samples = []
    # Example selection strategy: randomly select samples
    random.shuffle(data)
    informative_samples = data[:num_samples]
    return informative_samples

# Select the most informative samples
informative_samples = select_informative_samples(unlabeled_data, num_samples=3)

# Print the selected samples
for sample in informative_samples:
    print(sample)

# Define a function to label informative samples
def label_samples(samples):
    labeled_samples = []
    for sample in samples:
        print(sample)
        label = input("Enter the label (I-ENTITY, O): ")
        labeled_samples.append((sample, label))
    return labeled_samples

# Label the informative samples
labeled_samples = label_samples(informative_samples)

# Define a function to convert labeled samples to the required format
def convert_to_dataset(samples):
    dataset = []
    for sample in samples:
        tokens = sample[0].split()
        labels = sample[1].split()
        sentence = []
        for i in range(len(tokens)):
            sentence.append((tokens[i], labels[i]))
        dataset.append(sentence)
    return dataset

# Convert the labeled samples to the required format
dataset = convert_to_dataset(labeled_samples)

# Fine-tune the XLM-RoBERTa NER model with the labeled data
ner_pipeline = pipeline("ner", model="xlm-roberta-large-finetuned-conll02-spanish", tokenizer="xlm-roberta-large-finetuned-conll02-spanish", grouped_entities=True)
ner_pipeline.train(dataset, learning_rate=0.0001, mini_batch_size=8, max_epochs=10)

# Example sentence to test the updated NER model
text = "저는 David Kim 입니다. My phone number is 555-1234, and my account number is 123456789."

# Use the updated NER model to extract named entities from the test sentence
named_entities = ner_pipeline(text)

# Print the named entities
for entity in named_entities:
    print(entity)

# Evaluate the performance of the updated NER model using the test dataset
test_dataset = convert_to_dataset(test_data)
results = ner_pipeline.evaluate(test_dataset)
print(results)

# Save the updated NER model to a file
model_path = "korean_ner_model"
ner_pipeline.save(model_path)

# Load the saved NER model
loaded_ner_pipeline = pipeline("ner", model=model_path, tokenizer="xlm-roberta-large-finetuned-conll02-spanish")

# Example sentence to test the loaded NER model
text = "내 전화번호는 010-1234-5678입니다. My name is Jane Doe, and I work at ABC Corporation."

# Use the loaded NER model to extract named entities from the test sentence
named_entities = loaded_ner_pipeline(text)

# Print the named entities
for entity in named_entities:
    print(entity)

# Load the saved NER model
loaded_ner_pipeline = pipeline("ner", model=model_path, tokenizer="xlm-roberta-large-finetuned-conll02-spanish")

# Example sentence to test the loaded NER model
text = "카드번호는 1234-5678-9012-3456입니다. My name is John Smith, and I live in Seoul."

# Use the loaded NER model to extract named entities from the test sentence
named_entities = loaded_ner_pipeline(text)

# Print the named entities
for entity in named_entities:
    print(entity)

# Prompt the user for feedback on the extracted named entities
feedback = input("Enter feedback (Y/N): ")

# Update the NER model based on the feedback
if feedback == "Y":
    corrected_entities = input("Enter corrected entities (e.g. 1234-5678-9012-3456=CREDIT_CARD_NUMBER): ")
    corrected_entities = corrected_entities.split("=")
    corrected_named_entities = []
    for entity in named_entities:
        if entity["word"] in corrected_entities[0]:
            corrected_named_entities.append({"entity": corrected_entities[1], "start": entity["start"], "end": entity["end"], "score": entity["score"]})
        else:
            corrected_named_entities.append(entity)
    labeled_sample = (text, " ".join([entity["entity"] for entity in corrected_named_entities]))
    corrected_dataset = convert_to_dataset([labeled_sample])
    loaded_ner_pipeline.train(corrected_dataset, learning_rate=0.0001, mini_batch_size=8, max_epochs=10)


    
    
