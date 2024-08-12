import spacy
import json

def load_spacy_model():
    try:
        return spacy.load("en_core_web_md")
    except OSError:
        print("Model 'en_core_web_md' not found. Downloading...")
        spacy.cli.download("en_core_web_md")
        return spacy.load("en_core_web_md")

def get_named_entities(text, nlp, entity_type):
    doc = nlp(text)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    return entities

def main():
    nlp = load_spacy_model()

    print("Welcome to the spaCy NER Chatbot!")
    print("Paste your text and I'll extract named entities for you.")

    while True:
        text = input("\nPaste your text (or type 'quit' to exit): ")
        if text.lower() == 'quit':
            break

        entities = get_named_entities(text, nlp, "")

        print("\nExtracted entities:")
        for entity in entities:
            print(f"{entity['text']} - {entity['label']}")

if __name__ == "__main__":
    main()
