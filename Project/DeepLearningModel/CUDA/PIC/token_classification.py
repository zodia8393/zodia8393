import torch
import transformers

# Check if a GPU is available and use it if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define constants
MODEL_NAME = "dslim/bert-base-NER"
MAX_LEN = 128

# Load model and tokenizer
model = transformers.AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_text(text):
    """
    Tokenizes input text using the tokenizer.
    """
    tokenized_text = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_token_type_ids=False,
        return_attention_mask=True,
        return_tensors="pt"
    )
    return tokenized_text

def censor_entities(text, entity_labels):
    """
    Censors the entities in the input text using the NER model.
    """
    tokenized_text = tokenize_text(text)
    input_ids = tokenized_text["input_ids"].to(device)
    attention_mask = tokenized_text["attention_mask"].to(device)
    
    with torch.no_grad():
        output = model(input_ids, attention_mask)
    
    label_indices = torch.argmax(output[0], axis=2)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    output_tokens = []
    for token, label_idx in zip(tokens, label_indices[0]):
        label = model.config.id2label[label_idx]
        if label in entity_labels:
            token = "[CENSORED]"
        output_tokens.append(token)
    
    censored_text = tokenizer.convert_tokens_to_string(output_tokens)
    return censored_text

# Test the functions
text = "John Smith's phone number is 555-1234 and his email is john.smith@example.com."
entity_labels = ["PER", "PHONE", "EMAIL"]
censored_text = censor_entities(text, entity_labels)

print(f"Original text: {text}")
print(f"Censored text: {censored_text}")
