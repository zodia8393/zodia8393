import pandas as pd
from text_classification import censor_entities

# Load the model and tokenizer
model_name = "dslim/bert-base-NER"
model = transformers.AutoModelForTokenClassification.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

# Check if a GPU is available and use it if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the data and preprocess it
data = pd.read_csv("data.csv")
data = data.fillna("")
data["text"] = data["text"].apply(lambda x: x.replace("\n", " ").strip())
data["censored_text"] = data["text"].apply(lambda x: censor_entities(x, ["PER", "PHONE", "EMAIL"]))

# Save the censored data
data.to_csv("censored_data.csv", index=False)
