import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("The patient presents with acute myocardial infarction.")
tokens = [token.text for token in doc]
print(token)
