TCR SOURCE CLASSIFICATION PROJECT (TF-IDF + LOGISTIC REGRESSION)

======================================================================

OVERVIEW

This project builds a machine learning system that classifies T Cell Receptor (TCR) sequences into one of four biological source categories:

viral
bacterial
cancer
autoimmune

The model uses:

CDR3β amino acid sequences
Vβ and Jβ gene annotations (when available)
Character-level TF-IDF features
Logistic Regression classifier
Stratified cross-validation
Holdout validation

The goal is to predict the biological origin of immune responses from sequence patterns alone.

======================================================================

PROBLEM SUMMARY

Each TCR sequence represents a receptor from the immune system. These receptors recognize antigens from viruses, bacteria, tumors, or self-antigens.

Task:
Predict the source category of a TCR using sequence + optional gene information.

======================================================================

TOOLS USED (WHAT EACH ONE DOES)

PYTHON
Main programming language used for the entire project.

PANDAS
Loads CSV files
Handles tabular data
Used for cleaning labels and building datasets

NUMPY
Handles numerical arrays
Stores feature matrices and model outputs

REGEX (re)
Cleans amino acid sequences
Removes invalid characters

SCIKIT-LEARN (sklearn)
Machine learning library used for:

* TfidfVectorizer → converts sequences into numeric features
* LogisticRegression → classification model
* StratifiedKFold → cross-validation
* train_test_split → validation split
* f1_score → evaluation metric

======================================================================

TF-IDF (CHARACTER N-GRAMS)

Converts sequences into numerical features by breaking them into overlapping character chunks.

Example:
CDR3: CASSLGQETQYF
→ "CAS", "ASS", "SSL", "SLG", ...

This captures local biological motifs in sequences.

======================================================================

DATA DESCRIPTION

INPUT FILES:

TCR-Processed-Raw.csv (training)
test_set.csv (testing)

IMPORTANT COLUMNS:

CDR3.beta.aa
Main amino acid sequence used for prediction.

TRBV
V gene segment (optional signal).

TRBJ
J gene segment (optional signal).

Pathology
Raw biological label (needs cleaning).

======================================================================

LABEL CLEANING

Raw labels are messy, so they are mapped into 4 classes:

viral → influenza, CMV, HIV, EBV, SARS, etc.
bacterial → tuberculosis, E. coli, etc.
cancer → melanoma, tumor, TAA, etc.
autoimmune → diabetes, Parkinson’s, self-reactive diseases

Unclear labels are removed.

======================================================================

FEATURE ENGINEERING

Each sample is converted into a single text string:

CDR3 sequence + repeated gene signals + length token

Example:
CASSLGQETQYF VTRBV VTRBV VTRBV JTRBJ JTRBJ JTRBJ LEN

Why this works:

Repeating genes increases their importance
Length adds biological context
Combines sequence + immune information

======================================================================

MODEL

LOGISTIC REGRESSION

A linear classifier that:

Works well with sparse TF-IDF data
Is fast and stable
Handles multi-class classification

Settings:

class_weight="balanced" (handles imbalance)
multi_class="multinomial"

======================================================================

CROSS VALIDATION

Stratified K-Fold (5 splits)

Splits dataset into 5 parts
Keeps class distribution balanced
Trains and validates across all folds

Final score = average performance

======================================================================

HOLDOUT VALIDATION

80% training
20% validation

Used to check real-world performance and prevent overfitting.

======================================================================

FINAL PIPELINE

Load dataset
Clean labels
Extract sequences
Build feature strings
Convert to TF-IDF vectors
Train Logistic Regression model
Run cross-validation
Run holdout validation
Train final model on full data
Predict test set

======================================================================

PROJECT PROGRESSION (COMMITS EXPLAINED)

FIRST COMMIT – CONNECTIVITY TEST
Initial repository setup and connectivity check
Ensured GitHub repo was linked and working correctly

BUILD STARTING CODE (PROMPT GENERATED)
Created the first full working main.py using a prompt
Set up baseline pipeline structure:
data loading → preprocessing → TF-IDF → model training

FEATURE ENGINEERING / MODEL IMPROVEMENT
increased max_features → 100000
changed ngram_range → (3,6)
Improved feature representation for biological patterns
Made model capture richer sequence motifs

FINAL TF-IDF OPTIMIZATION
changed analyzer = "char" instead of "char_wb"
Improved raw character-level pattern capture
Better handling of full sequence boundaries

======================================================================

LIMITATIONS

TF-IDF cannot capture deep biological structure
No protein language model used
Gene annotations are missing in some samples
Label mapping is rule-based (not perfect)
Cannot model long-range biological interactions

======================================================================

ASSUMPTIONS

Pathology labels are mostly correct
CDR3 sequence is sufficient signal
Missing gene data does not heavily bias results
Train/test distributions are similar