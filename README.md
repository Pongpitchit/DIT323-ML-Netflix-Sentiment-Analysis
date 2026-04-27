# DIT323 ML — Netflix Sentiment Analysis

This project runs an **advanced Machine Learning sentiment analysis pipeline** on a Netflix dataset and produces a **dashboard image**.

## Files

- `Netflix_Dataset.csv`: input dataset (must be in the same folder as the script)
- `netflix_sentiment_pipeline.py`: main sentiment analysis pipeline script
- `netflix_pililine.py`: an alternative/extended pipeline script (same dataset/goal)
- `netflix_ml_dashboard.png`: output dashboard image (generated after running)

## Requirements

Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn textblob scipy
```

Notes:
- The scripts use `matplotlib.use("Agg")`, so they **do not require a GUI** and will save plots to files.

## How to run

From the project folder:

```bash
python netflix_sentiment_pipeline.py
```

Or run the other pipeline script:

```bash
python netflix_pililine.py
```

## Output

After a successful run, you should see (at least):

- `netflix_ml_dashboard.png`: a summary dashboard of model performance and dataset statistics

## Troubleshooting

- **File not found `Netflix_Dataset.csv`**: keep `Netflix_Dataset.csv` in the same directory as the script, or update `CSV_PATH = "Netflix_Dataset.csv"` inside the script.
- **TextBlob errors**: if `textblob` is installed but fails, try upgrading:

```bash
pip install -U textblob
```
