# Sentiment Social Media Analysis

This project trains and evaluates sentiment analysis models on social media text, then uses the best model to predict sentiment labels.

## Project Structure

- `Data/`
  - `Twitter_Data.csv`
  - `Reddit_Data.csv`
- `Notebook/`
  - `Sentiment.ipynb`

## What This Project Does

1. Loads and cleans social media comments.
2. Preprocesses text (cleaning, tokenization, lemmatization).
3. Converts text to TF-IDF features.
4. Trains multiple classifiers:
   - Logistic Regression
   - Naive Bayes
   - Random Forest
5. Evaluates models using:
   - Accuracy
   - Macro F1 score
   - Classification report
   - Confusion matrix
6. Selects the best model and saves:
   - `best_sentiment_model.pkl`
   - `tfidf_vectorizer.pkl`
7. Runs predictions on sample text and full dataset rows.

## Requirements

Install Python packages:

```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn imbalanced-learn joblib
```

Download required NLTK resources (run once):

```python
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
```

## How To Run

1. Open `Notebook/Sentiment.ipynb`.
2. Run cells from top to bottom.
3. Make sure the training/evaluation cell completes before prediction cells.
4. Check generated output files in `Notebook/`:
   - `best_sentiment_model.pkl`
   - `tfidf_vectorizer.pkl`

## Notes

- The notebook currently loads a subset (`nrows=10000`) in `load_data` for faster experimentation.
- You can remove or increase that limit for full-dataset training.
- If you get NLTK `LookupError`, run the NLTK download commands above.

## Future Improvements

- Add a `requirements.txt` file for reproducible installs.
- Add cross-validation and hyperparameter tuning.
- Add a script version of the notebook pipeline for automation.
