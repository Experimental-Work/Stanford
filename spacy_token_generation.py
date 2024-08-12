import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Define the text
text = "The patient presents with acute myocardial infarction."

# Print the original text
print("Original text:")
print(text)

# Process the text and generate tokens
doc = nlp(text)
tokens = [(i, token.text) for i, token in enumerate(doc)]

# Print the tokens with their indices
print("\nTokens (index: token):")
for index, token in tokens:
    print(f"{index}: {token}")
