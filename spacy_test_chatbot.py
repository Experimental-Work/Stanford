import spacy
import json

def load_spacy_model():
    return spacy.load("en_core_web_sm")

def get_named_entities(text, nlp, entity_type):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ == entity_type]
    return entities

def main():
    nlp = load_spacy_model()

    print("Welcome to the spaCy NER Chatbot!")
    print("Paste your text and I'll extract named entities for you.")

    while True:
        text = input("\nPaste your text (or type 'quit' to exit): ")
        if text.lower() == 'quit':
            break

        entity_type = input("Enter the entity type you want to extract (e.g., PERSON, ORG, GPE): ").upper()

        entities = get_named_entities(text, nlp, entity_type)

        result = {
            "entity_type": entity_type,
            "entities": entities
        }

        print("\nExtracted entities:")
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
