import os
import sys
import json
import pickle
import argparse
from nltk.util import ngrams

# Helper functions copied from my notebook
def get_perplexity(model, test_data):
    test_ngrams = []
    for method in test_data:
        processed = list(model.vocab.lookup(method))
        method_ngrams = ngrams(processed, model.order, 
                               pad_left=True, left_pad_symbol="<s>",
                               pad_right=True, right_pad_symbol="</s>")
        test_ngrams.extend(method_ngrams)
    return model.perplexity(test_ngrams)

def get_predictions_for_method(model, tokens):
    results = []
    n = model.order
    safe_tokens = list(model.vocab.lookup(tokens))
    padded = ["<s>"] * (n - 1) + safe_tokens
    
    for i in range(n - 1, len(padded)):
        context = tuple(padded[i - (n - 1) : i])
        best_token = model.generate(1, text_seed=context)
        pred_prob = model.score(best_token, context)
        ground_truth = padded[i]
        
        results.append({
            "context": list(context),
            "predToken": best_token,
            "predProbability": round(pred_prob, 4),
            "groundTruth": ground_truth
        })
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate N-gram model on tokenized Java methods.")
    parser.add_argument("input_file", help="Path to the .txt file containing tokenized Java methods.")
    parser.add_argument("--model", default="best_ngram_model.pkl", help="Path to the pickled model file.")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: File '{args.input_file}' not found.")
        sys.exit(1)
        
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found. Please run the notebook first to generate it.")
        sys.exit(1)

    # Load the text file
    print(f"Loading data from {args.input_file}...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        test_data = [line.strip().split() for line in f if line.strip()]

    # Load the pickled model
    print(f"Loading model from {args.model}...")
    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    # Generate predictions and format the JSON
    print(f"Evaluating {len(test_data)} methods... this might take a moment.")
    total_perp = get_perplexity(model, test_data)
    
    results = {
        "testSet": os.path.basename(args.input_file),
        "perplexity": round(total_perp, 2),
        "data": []
    }
    
    for i, method_tokens in enumerate(test_data):
        predictions = get_predictions_for_method(model, method_tokens)
        results["data"].append({
            "index": f"ID{i+1}",
            "tokenizedCode": " ".join(method_tokens),
            "contextWindow": model.order,
            "predictions": predictions
        })
        
    # Save the output
    output_filename = f"results-{os.path.basename(args.input_file).replace('.txt', '.json')}"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
        
    print(f"Success! Output saved to {output_filename}")

if __name__ == "__main__":
    main()