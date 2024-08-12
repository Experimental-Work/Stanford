import spacy
import numpy as np

def load_model(model_name):
    try:
        return spacy.load(model_name)
    except OSError:
        print(f"Model {model_name} not found. Downloading...")
        spacy.cli.download(model_name)
        return spacy.load(model_name)

def process_text(nlp, text):
    doc = nlp(text)
    print(f"\nModel: {nlp.meta['name']}")
    print(f"Vector dimension: {nlp.vocab.vectors.shape[1]}")
    
    print("\nTokens (index: token):")
    for i, token in enumerate(doc):
        print(f"{i}: {token.text}")
    
    print("\nToken vectors:")
    for token in doc:
        vector = [round(float(x), 4) for x in token.vector]
        print(f"{token.text}: {vector}")
    
    doc_vector = doc.vector
    mean_vector = [round(float(x), 4) for x in doc_vector]
    print("\nMean document vector:")
    print(mean_vector)

def main():
    # Define the text
    text = "The patient presents with acute myocardial infarction."
    
    print("Original text:")
    print(text)
    
    # Load and process with all three models
    models = ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]
    
    for model_name in models:
        nlp = load_model(model_name)
        process_text(nlp, text)
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()
