# Replication Guide: PEAD.txt (Meursault et al. 2022)

This document describes the data requirements and steps to replicate the core regression from the paper. All data access assumes a WRDS (Wharton Research Data Services) subscription.

## 1. Data Sources

### 1.1 Earnings Call Transcripts

- **Database:** Capital IQ Transcripts (via WRDS)
- **What to pull:** Full text of quarterly earnings conference calls, including:
  - Management presentation section
  - Q&A section
  - Transcript metadata: company identifier, date, transcript version (preliminary vs. final), upload timestamp
- **Sample period:** 2008Q1 through 2019Q4 (training starts from 2008; test predictions begin 2010Q1)
- **Expected size:** ~108,704 transcripts before filtering

### 1.2 Stock Returns (CRSP)

- **Database:** CRSP Daily Stock File (via WRDS)
- **What to pull:**
  - Daily stock returns (RET) and holding period returns
  - Share price (PRC), shares outstanding (SHROUT)
  - PERMNO identifiers
- **Purpose:** Compute one-day abnormal returns for the target variable and long-run CARs for the drift analysis

### 1.3 Financial Data (Compustat)

- **Database:** Compustat Quarterly (via WRDS)
- **What to pull:**
  - Quarterly earnings (EPSPXQ or IBQ)
  - Earnings announcement dates (RDQ)
  - Fiscal quarter identifiers
  - Firm size, book value
- **Purpose:** Construct classic SUE measures and control variables
- **Linking:** Use the Compustat/CRSP Merged dataset available on WRDS

### 1.4 Analyst Forecasts (IBES)

- **Database:** IBES Summary Statistics and Detail files (via WRDS, provided by Refinitiv)
- **What to pull:**
  - Median analyst EPS forecast (consensus forecast)
  - Actual reported EPS
  - Forecast period end date
- **Purpose:** Construct the main SUE measure (SUE3) as the analyst-forecast-based earnings surprise
- **Linking:** Use the CRSP-to-IBES linking table (see Freda Song Drechsler's script at fredasongdrechsler.com/full-python-code/iclink)

### 1.5 Factor Returns

- **Fama-French factors:** Download from Ken French's data library (mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
  - Fama-French 3 factors + momentum (for abnormal return calculation)
  - Fama-French 5 factors + momentum (for alpha tests)
  - Six size and book-to-market portfolio returns (for benchmark CARs)
- **q5 factors:** Download from global-q.org (for robustness tests)

## 2. Dataset Construction

### 2.1 Merging Datasets

1. Start with Capital IQ transcripts. Extract the company identifier (CIQ company ID).
2. Use the identifier crosswalk provided with the Transcripts dataset to link to CRSP PERMNO.
3. Link to IBES using the CRSP-IBES linking table.
4. Link to Compustat via the Compustat/CRSP Merged dataset on WRDS.
5. Winsorize all continuous variables at the 1% and 99% levels.

### 2.2 Determining the Earnings Call Day Return

The Capital IQ Transcripts database does not include the exact call time but does record when the transcript was created. Use this heuristic:

- If the first version is marked as **preliminary** and was uploaded **before 3:00 PM ET** on date t: use the return between t-1 and t.
- If the first version is marked as **final (edited)** and was uploaded sometime during date t: use the return between t-1 and t.
- Otherwise: use the return between t and t+1.

This ensures the return window covers the actual call time and avoids using future information.

### 2.3 Computing One-Day Abnormal Returns

Use the **WRDS Event Studies tool** to compute one-day abnormal returns with:

- Model: Fama-French 3-factor + Carhart momentum
- Default estimation window, number of valid returns, and gap parameters
- The abnormal return isolates the "unexpected" portion of the stock return on the earnings call day

Alternatively, compute manually:

```
AR_{f,t} = R_{f,t} - E[R_{f,t}]
```

where E[R] is the predicted return from a Fama-French 3-factor + momentum regression estimated over a window preceding the event.

### 2.4 Categorizing Returns

Split one-day abnormal returns into three categories:

1. **Flat:** The 33% of observations closest to zero (first tercile of absolute abnormal returns). Cutoffs are based on the training set.
2. **High:** Positive abnormal returns not in the flat category.
3. **Low:** Negative abnormal returns not in the flat category.

This produces a roughly even three-way split.

## 3. Text Processing

### 3.1 Preprocessing Steps

Starting from the raw transcript text:

1. **Separate** the management presentation section from the Q&A section.
2. **Lowercase** all words.
3. **Replace numbers** with a generic token "#". For example:
   - "$1000.00" becomes "$#"
   - "Q3" becomes "Q#"
4. **Remove stopwords** using the Snowball stemmer's stopword list (snowballstem.org). This removes common words like "the," "is," "at," etc.
5. **Do not stem** the remaining words. No other word processing is applied.

### 3.2 Feature Construction

1. Compute **unigram** (single word) and **bigram** (two-word) frequencies for each document.
2. Select the **1,000 most common unigrams** and **1,000 most common bigrams** from the **presentation** section.
3. Select the **1,000 most common unigrams** and **1,000 most common bigrams** from the **Q&A** section.
4. This gives **4,000 features** total. Feature selection is based on the training set and thus varies across rolling windows.
5. Transform raw counts to log frequencies:

```
x_{n,j} = log(1 + freq(j, n))
```

where freq(j, n) is the frequency of term j in document n.

## 4. Running the First Regression

### 4.1 Model Specification

The model is a **multinomial logistic regression with elastic net regularization** (Zou and Hastie, 2005). For each return category r in {H, F, L}:

```
log-odds(r) = log[ Pr(R=r | X=x) / Pr(R!=r | X=x) ] = beta_0r + beta_r^T * x
```

The objective function combines the multinomial log-likelihood with an elastic net penalty:

```
minimize:  -LogLikelihood + lambda * [ (1-alpha)/2 * ||beta||_2^2 + alpha * ||beta||_1 ]
```

**Hyperparameters:**
- alpha = 0.5 (fixed; balances L1 and L2 penalties equally)
- lambda: chosen by 10-fold cross-validation within each training window

### 4.2 Rolling Window Estimation

The model uses a **sliding window** approach to ensure strictly out-of-sample predictions:

| Iteration | Training Set    | Test Set |
|-----------|-----------------|----------|
| 1         | 2008Q1 - 2009Q4 | 2010Q1   |
| 2         | 2008Q2 - 2010Q1 | 2010Q2   |
| 3         | 2008Q3 - 2010Q2 | 2010Q3   |
| ...       | ...             | ...      |
| 40        | 2017Q4 - 2019Q3 | 2019Q4   |

At each iteration:
1. Select the 8 most recent quarters as the training set.
2. Select the 1,000 most common unigrams and bigrams (separately for presentation and Q&A) **from the training set**.
3. Compute log-frequency features for the training set.
4. Compute return tercile cutoffs (for the flat/high/low split) **from the training set**.
5. Fit the elastic net logistic regression with 10-fold CV to choose lambda.
6. Predict log-odds for each category on the test quarter.
7. Compute SUE.txt for each test-quarter observation.

### 4.3 Computing SUE.txt

From the model's predicted log-odds on the test set:

```
SUE.txt = log-odds(H) - log-odds(L)
```

- SUE.txt > 0: the call text indicates good news
- SUE.txt < 0: the call text indicates bad news
- SUE.txt = 0: no unexpected information

### 4.4 Implementation in R

The paper uses the `glmnet` package in R (Friedman, Hastie, and Tibshirani, 2010).

```r
library(glmnet)
library(tm)

# --- Step 1: Text processing ---

preprocess_transcript <- function(text) {
  text <- tolower(text)
  # Replace all numbers with #
  text <- gsub("[0-9]+\\.?[0-9]*", "#", text)
  return(text)
}

build_dtm <- function(corpus, ngram_range = c(1, 1), top_n = 1000) {
  tokenizer <- function(x) {
    if (ngram_range[2] == 2) {
      # bigram tokenizer
      unlist(lapply(NLP::ngrams(words(x), 2), paste, collapse = " "))
    } else {
      words(x)
    }
  }

  dtm <- DocumentTermMatrix(
    corpus,
    control = list(
      tokenize = tokenizer,
      stopwords = TRUE,       # Snowball English stopwords
      weighting = weightTf
    )
  )

  # Select top_n most frequent terms
  freq <- colSums(as.matrix(dtm))
  top_terms <- names(sort(freq, decreasing = TRUE))[1:top_n]
  dtm <- dtm[, top_terms]

  # Log transform
  dtm_log <- log(1 + as.matrix(dtm))
  return(dtm_log)
}

# --- Step 2: Build feature matrix ---
# For each rolling window, build 4 DTMs:
# - presentation unigrams (top 1000)
# - presentation bigrams (top 1000)
# - Q&A unigrams (top 1000)
# - Q&A bigrams (top 1000)
# Concatenate into a single 4000-column feature matrix X

# --- Step 3: Fit the model ---

fit_model <- function(X_train, y_train) {
  # y_train should be a factor with levels c("H", "F", "L")
  cv_fit <- cv.glmnet(
    x = X_train,
    y = y_train,
    family = "multinomial",
    alpha = 0.5,           # elastic net mixing parameter
    nfolds = 10,
    type.measure = "deviance"
  )
  return(cv_fit)
}

# --- Step 4: Predict and compute SUE.txt ---

compute_sue_txt <- function(cv_fit, X_test) {
  # Predict log-odds for each category
  preds <- predict(cv_fit, newx = X_test, s = "lambda.min", type = "link")
  # preds is an array with columns for each class

  log_odds_H <- preds[, , "H"]
  log_odds_L <- preds[, , "L"]

  sue_txt <- log_odds_H - log_odds_L
  return(sue_txt)
}

# --- Step 5: Rolling window loop ---

quarters <- seq(as.Date("2010-01-01"), as.Date("2019-10-01"), by = "quarter")
all_sue_txt <- list()

for (q in seq_along(quarters)) {
  test_quarter <- quarters[q]

  # Define 8-quarter training window ending before test_quarter
  # ... (select training and test data)

  # Build features from training set vocabulary
  # ... (call build_dtm for each section x ngram combination)

  # Compute return tercile cutoffs from training set
  # ... (categorize returns into H, F, L)

  # Fit model
  cv_fit <- fit_model(X_train, y_train)

  # Predict on test quarter
  sue_txt <- compute_sue_txt(cv_fit, X_test)
  all_sue_txt[[q]] <- data.frame(
    permno = test_data$permno,
    quarter = test_quarter,
    sue_txt = sue_txt
  )
}

results <- do.call(rbind, all_sue_txt)
```

### 4.5 Implementation in Python (Alternative)

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV

def preprocess(text):
    import re
    text = text.lower()
    text = re.sub(r'\d+\.?\d*', '#', text)
    return text

def build_features(docs_train, docs_test, max_features=1000, ngram_range=(1,1)):
    """Build log-frequency features using training vocabulary."""
    vec = CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english'
    )
    X_train = vec.fit_transform(docs_train)
    X_test = vec.transform(docs_test)
    # Log transform
    X_train = np.log1p(X_train.toarray())
    X_test = np.log1p(X_test.toarray())
    return X_train, X_test

def run_rolling_window(data, quarters):
    """
    data: DataFrame with columns:
      - permno, quarter, presentation_text, qa_text, abnormal_return
    quarters: list of test quarter dates
    """
    results = []

    for test_q in quarters:
        # 8-quarter training window
        train_quarters = [q for q in all_quarters if q < test_q][-8:]
        train = data[data['quarter'].isin(train_quarters)]
        test = data[data['quarter'] == test_q]

        # Categorize training returns into H/F/L
        abs_ar = train['abnormal_return'].abs()
        tercile_cutoff = abs_ar.quantile(1/3)
        y_train = pd.cut(
            train['abnormal_return'],
            bins=[-np.inf, -tercile_cutoff, tercile_cutoff, np.inf],
            labels=['L', 'F', 'H']
        )

        # Build 4 feature sets
        feat_pairs = [
            ('presentation_text', (1,1), 1000),  # pres unigrams
            ('presentation_text', (2,2), 1000),  # pres bigrams
            ('qa_text', (1,1), 1000),            # qa unigrams
            ('qa_text', (2,2), 1000),            # qa bigrams
        ]

        X_train_parts, X_test_parts = [], []
        for col, ngram, nfeat in feat_pairs:
            docs_tr = train[col].apply(preprocess).tolist()
            docs_te = test[col].apply(preprocess).tolist()
            Xtr, Xte = build_features(docs_tr, docs_te, nfeat, ngram)
            X_train_parts.append(Xtr)
            X_test_parts.append(Xte)

        X_train_all = np.hstack(X_train_parts)
        X_test_all = np.hstack(X_test_parts)

        # Fit elastic net logistic regression
        # sklearn uses C = 1/lambda; search over a grid
        model = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            l1_ratio=0.5,   # equivalent to alpha=0.5
            multi_class='multinomial',
            max_iter=5000
        )

        param_grid = {'C': np.logspace(-4, 2, 20)}
        cv = GridSearchCV(model, param_grid, cv=10, scoring='neg_log_loss')
        cv.fit(X_train_all, y_train)

        # Predict log-odds on test set
        best_model = cv.best_estimator_
        log_probs = best_model.predict_log_proba(X_test_all)
        classes = list(best_model.classes_)
        idx_H = classes.index('H')
        idx_L = classes.index('L')

        # SUE.txt = log-odds(H) - log-odds(L)
        # log-odds(r) = log[P(r) / (1-P(r))]
        probs = best_model.predict_proba(X_test_all)
        log_odds_H = np.log(probs[:, idx_H] / (1 - probs[:, idx_H]))
        log_odds_L = np.log(probs[:, idx_L] / (1 - probs[:, idx_L]))
        sue_txt = log_odds_H - log_odds_L

        for i, row in test.iterrows():
            results.append({
                'permno': row['permno'],
                'quarter': test_q,
                'sue_txt': sue_txt[test.index.get_loc(i)]
            })

    return pd.DataFrame(results)
```

## 5. Verification Checks

After computing SUE.txt for the full sample, verify the following:

1. **Distribution:** SUE.txt should be roughly symmetric around zero with a standard deviation of approximately 1-2.
2. **Correlation with SUE:** SUE.txt and classic SUE should have a moderate positive correlation (they measure related but distinct quantities).
3. **Quintile spread returns:** Sort observations into SUE.txt quintiles each quarter. The top-minus-bottom quintile spread in 63-day CARs should be approximately 2.87%.
4. **Panel regression:** Regress 63-day CAR on standardized SUE.txt (and optionally SUE) with firm and year-quarter fixed effects. The coefficient on SUE.txt should be positive and significant, with a normalized magnitude of 3-6%.

## 6. File Checklist

Before starting, confirm access to:

- [ ] WRDS account with Capital IQ Transcripts access
- [ ] WRDS CRSP Daily Stock File
- [ ] WRDS Compustat Quarterly
- [ ] WRDS IBES Summary/Detail
- [ ] WRDS Compustat/CRSP Merged
- [ ] Fama-French factor data (downloadable without WRDS)
- [ ] R with `glmnet` and `tm` packages, or Python with `scikit-learn`
- [ ] CRSP-IBES linking table (fredasongdrechsler.com/full-python-code/iclink)
