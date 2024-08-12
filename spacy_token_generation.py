import spacy
import numpy as np

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Define the text
text = "The patient presents with acute myocardial infarction."

# Print the original text
print("Original text:")
print(text)

# Process the text and generate tokens
doc = nlp(text)

# Print the tokens with their indices
print("\nTokens (index: token):")
for i, token in enumerate(doc):
    print(f"{i}: {token.text}")

# Print vector information
print(f"\nVector dimension: {nlp.vocab.vectors.shape[1]}")

# Print the vector representation of each token
print("\nToken vectors:")
for token in doc:
    # Convert the vector to a list and round each value to 4 decimal places
    vector = [round(float(x), 4) for x in token.vector]
    print(f"{token.text}: {vector}")

# Optional: Print the mean vector of the entire document
doc_vector = doc.vector
mean_vector = [round(float(x), 4) for x in doc_vector]
print("\nMean document vector:")
print(mean_vector)
