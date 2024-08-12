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
tokens = [token.text for token in doc]

# Print the tokens
print("\nTokens:")
print(tokens)
