import pandas as pd
from openpyxl import load_workbook
from langdetect import detect

# Load the Excel file
df = pd.read_excel("file.xlsx")

# Create new columns for English and Korean
df["English"] = ""
df["Korean"] = ""

# Iterate through each row of the dataframe
for i, row in df.iterrows():
    text = row["text"]
    # Detect the language of the text
    language = detect(text)
    # Store the text in the appropriate column
    if language == "en":
        df.at[i, "English"] = text
    elif language == "ko":
        df.at[i, "Korean"] = text

# Save the dataframe to the Excel file
book = load_workbook("file.xlsx")
writer = pd.ExcelWriter("file.xlsx", engine="openpyxl") 
writer.book = book
df.to_excel(writer, index=False)
writer.save()
