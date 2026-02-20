# CSCI 455/555: GenAI for SD - Assignment 1
## Recommending Code Tokens via N-gram Models

**Author:** Nathaniel Callabresi  
**Course:** CSCI 455/555: GenAI for Software Development  
**Instructor:** Prof. Antonio Mastropaolo  

---

## Project Overview

This repository contains an end-to-end pipeline for predicting Java code tokens using N-gram language models. The project is divided into two main parts within the primary Jupyter Notebook (Callabresi_Assignment1.ipynb):

* **Part 1: MSR Pipeline (Minilab 1)** - Scrapes top Java repositories from GitHub (>4000 stars, no forks), extracts methods using javalang, and applies rigorous filtering (non-ASCII removal, 10-token minimum, 512-token maximum, minimum 10 unique tokens, and deduplication). It then tokenizes the code and splits it into discrete text files.
* **Part 2: Model Training & Evaluation** - Uses the NLTK library to train Lidstone-smoothed N-gram models ($n \in \{3, 5, 7\}$) on cumulative subsets of the mined data. It evaluates the models via perplexity, selects the best configuration, and generates structured JSON predictions.

---

## Installation & Dependencies

You can install the required dependencies in one of two ways:

### Option A: Using requirements.txt (Recommended)
Before running any code, install all required packages at once via the terminal:

    pip install -r requirements.txt

### Option B: Inline Notebook Installation
The Jupyter Notebook includes !pip install commands in the relevant cells to automatically install missing dependencies (like javalang, gitpython, pandas, lizard, matplotlib, and nltk) as it runs.

---

## How to Run the Pipeline (Jupyter Notebook)

To successfully execute the pipeline and ensure the provided test data is read correctly, you must follow these exact steps:

1. **Run the initial setup cells:** Start by running the first 10 cells of Callabresi_Assignment1.ipynb. Stop once you have executed the cell that creates the dataset/ngram_dataset/ directory and generates and saves the MSR split files.
2. **Upload the provided test set:** At this point, the notebook has created the following 5 text files inside the dataset/ngram_dataset/ directory:
   * train.txt
   * val.txt
   * test1.txt
   * test2.txt
   * test3.txt

   > **IMPORTANT:** Before proceeding to the Model Training (Part 2) cells, you must manually upload/place the provided_test.txt file into this exact dataset/ngram_dataset/ directory so it looks exactly like the rest of the generated files.

3. **Run the rest of the notebook:** Once the provided_test.txt file is in place, you can safely "Run All" or execute the remaining cells sequentially.

---

## Generated Files Explained

After running the complete notebook, the following critical files will be generated:

* **MSR Data Splits (test1.txt, test2.txt, test3.txt, train.txt, val.txt):** The raw, tokenized Java methods mined from GitHub, split into separate subsets.
* **metadata.json:** Tracks which GitHub repositories and files were used, dataset statistics, and instructions for replicating the splits.
* **best_ngram_model.pkl:** A serialized (pickled) version of the best-performing N-gram model. This is saved for use in the command-line interface.
* **results-xxxxxx.json:** The structured JSON output containing the context windows, predicted tokens, probabilities, and overall perplexity evaluated on the provided_test.txt dataset.
* **results-yyyyyy.json:** The structured JSON output evaluated on the self-created test set (test.txt / test1.txt data).

---

## Command Line Tool (evaluate.py)

In addition to the Jupyter Notebook, a standalone Python script (evaluate.py) is provided to evaluate the trained model on any tokenized .txt file via the command line.

### Usage

    python evaluate.py <input_file> [--model <model_path>]

### Arguments & Flags
* **input_file (Required):** The path to the .txt file containing the tokenized Java methods you want to test (e.g., dataset/ngram_dataset/provided_test.txt).
* **--model (Optional):** The path to the pickled model file. It defaults to best_ngram_model.pkl in the current directory.

### Example

    python evaluate.py dataset/ngram_dataset/provided_test.txt --model best_ngram_model.pkl

### CLI Output
The script will load the data and the model, calculate the perplexity, generate predictions, and output a JSON file named results-<your_input_filename>.json in the current working directory.

---

## Hyper-parameters Tuned

During the evaluation phase, several hyper-parameters were systematically tuned:

* **Context Window ($N$):** Evaluated $n \in \{3, 5, 7\}$.
  * Reasoning: The 3-gram model consistently achieved the lowest perplexity. Because the MSR pipeline pulled highly varied data from 700 different repositories, the vocabulary size exploded. Larger context windows ($n=5, 7$) suffered from severe data sparsity (too many completely unseen n-grams), lowering the model's confidence.
* **Smoothing Algorithm:** Lidstone Smoothing was implemented to handle unseen n-grams and prevent division-by-zero errors in perplexity calculations.
* **Lidstone Gamma ($\gamma$):** Set to 0.01.
  * Reasoning: Standard Laplace smoothing ($\gamma = 1$) takes too much probability mass away from observed n-grams, especially in tasks with massive, sparse vocabularies like source code. A much smaller gamma (0.01) successfully smoothed the zero-frequency n-grams without heavily penalizing the probabilities of known patterns.
* **Unknown Token Cutoff (UNK_LIMIT):** Set to 3.
  * Reasoning: Any token appearing fewer than 3 times in the training data was mapped to a `<UNK>` token. This dynamically filtered out obscure, project-specific variable names or typos, effectively mitigating the vocabulary explosion problem and improving generalizability.

---

## Evaluation Results

Across the 9 model configurations tested (3 dataset sizes $\times$ 3 $n$-values), the best performance was achieved using the smallest dataset subset with the smallest context window.

* **Best Configuration:** Training Set 1 ($T_1$, 15,000 methods) with an $n=3$ (3-gram) model.
* **Validation Perplexity:** 39.78
* **Provided Test Set Perplexity:** 35.38
* **Self-Created Test Set Perplexity:** 41.37
