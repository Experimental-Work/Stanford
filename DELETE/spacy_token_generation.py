import numpy as np
import spacy


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
    print(f"Model name: {nlp.meta['name']}")
    print(f"Vector dimension: {nlp.vocab.vectors.shape[1]}")
    
    print("\nTokens (index: token):")
    for i, token in enumerate(doc):
        print(f"{i}: {token.text}")
    
    print("\nToken vectors:")
    all_vectors = []
    for token in doc:
        vector = token.vector
        all_vectors.append(vector)
        vector_rounded = [round(float(x), 4) for x in vector]
        print(f"{token.text}: {vector_rounded}")
        
        # Print vector statistics
        print(f"  Min: {np.min(vector):.4f}")
        print(f"  Max: {np.max(vector):.4f}")
        print(f"  Mean: {np.mean(vector):.4f}")
        print(f"  Standard deviation: {np.std(vector):.4f}")
    
    doc_vector = doc.vector
    mean_vector = [round(float(x), 4) for x in doc_vector]
    print("\nMean document vector:")
    print(mean_vector)
    
    # Print overall statistics
    all_vectors = np.array(all_vectors)
    print("\nOverall vector statistics:")
    print(f"Global min: {np.min(all_vectors):.4f}")
    print(f"Global max: {np.max(all_vectors):.4f}")
    print(f"Global mean: {np.mean(all_vectors):.4f}")
    print(f"Global standard deviation: {np.std(all_vectors):.4f}")
    
    print("\nVectorization method:")
    if "word2vec" in nlp.meta['description'].lower():
        print("Word2Vec")
    elif "glove" in nlp.meta['description'].lower():
        print("GloVe")
    else:
        print("Unknown (check model description)")
    
    print(f"\nModel description: {nlp.meta['description']}")

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
