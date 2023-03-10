from preprocessing import read_data, preprocess_data, TextClassificationDataset, get_data_loaders
from token_classification import censor_entities
from tqdm import tqdm

# Test the preprocessing module
train_data = read_data('train.tsv')
train_data = preprocess_data(train_data)
print(train_data[0])

# Test the token_classification module
text = "John Smith's phone number is 555-1234 and his email is john.smith@example.com."
entity_labels = ["PER", "PHONE", "EMAIL"]
censored_text = censor_entities(text, entity_labels)
print(censored_text)

# Test the dataset and data loader modules
train_loader, dev_loader = get_data_loaders('train.tsv', 'dev.tsv')
for batch in tqdm(train_loader):
    pass
