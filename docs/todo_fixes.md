# PEAD.txt Replication: Required Fixes

Fixes needed to match Meursault et al. (2022 JFQA) exactly. Ordered by impact on results.

## Priority 1: Return Measurement

### 1a. Long-run CARs must use FF6 size/BM benchmark portfolios

**Current:** Raw buy-and-hold returns (BHAR), no benchmark subtraction.

**Paper (Eq. 2):** Abnormal returns are computed daily as firm return minus the return of one of six size/book-to-market matched portfolios (from Ken French's data library). CARs are cumulative products of these daily abnormal returns.

```
AR_{f,q,t} = R_{f,q,t} - R^b_{f,q,t}
CAR^S_t = prod_{t=S}^{E} (AR^S_t)
```

where R^b is the benchmark portfolio return matched by size and book-to-market.

**Fix:** Download FF 6 portfolio daily returns. Assign each firm-quarter to one of the six portfolios using NYSE breakpoints for size and book-to-market. Subtract the matched portfolio return from the firm return each day before cumulating.

**File:** `build_dataset_lean.py:217-224` and the inline AR computation script.

### 1b. Event-day timing heuristic

**Current:** Always uses the nearest trading day to the call date.

**Paper (Online Appendix A):** The earnings call day return depends on transcript upload timing:
- If the first version is **preliminary** and uploaded **before 3:00 PM ET** on day t: return = t-1 to t.
- If the first version is **final (edited)** and uploaded during day t: return = t-1 to t.
- Otherwise: return = t to t+1.

**Fix:** Parse the transcript upload timestamp and version type from the XML metadata. Implement the three-way heuristic.

**File:** `parse_transcripts.py` (extract version/upload time), AR computation scripts.

### 1c. CRSP sentinel return codes

**Current:** `pd.to_numeric(errors='coerce')` does not catch CRSP sentinel values.

**Paper:** Returns of -66, -77, -88, -99 are CRSP missing codes, not real returns.

**Fix:** After loading CRSP returns, filter out rows where `ret` is in {-66, -77, -88, -99}.

**File:** All scripts that load CRSP daily returns.

## Priority 2: Feature Construction

### 2a. Use all four feature blocks (4,000 features)

**Current:** 1,000 presentation unigrams only.

**Paper (Section II.B):** Four blocks, 1,000 features each:
1. Presentation unigrams (top 1,000)
2. Presentation bigrams (top 1,000)
3. Q&A unigrams (top 1,000)
4. Q&A bigrams (top 1,000)

Features are log frequencies: `x_{n,j} = log(1 + freq(j, n))`.

**Fix:** Load both presentation and Q&A text. Build four separate CountVectorizers per rolling window. Concatenate into a 4,000-column sparse feature matrix.

**File:** `precompute_features.py`, `run_regression_lean.py`.

### 2b. Rolling-window vocabulary selection

**Current:** Fixed vocabulary from 2008-2009 sample, applied to all years.

**Paper:** The 1,000 most common tokens are selected from the **training set** at each rolling window iteration. Vocabulary changes every quarter.

**Fix:** Remove the pre-computation step. Instead, build vocabulary inside the rolling-window loop from the current 8-quarter training set.

**File:** `precompute_features.py` (remove or restructure), regression scripts.

### 2c. Preserve the `#` number token

**Current:** `CountVectorizer`'s default `token_pattern=r"(?u)\b\w\w+\b"` requires 2+ characters, silently dropping the single-character `#` token.

**Paper:** Numbers are replaced with `#` so that the model captures number-context words like `$#`, `Q#`, `#%` without using actual values.

**Fix:** Set `token_pattern=r"(?u)\b\w+\b"` (allow single-character tokens) or `token_pattern=r"(?u)\b[\w#]+\b"`.

**File:** All CountVectorizer calls.

## Priority 3: Model Specification

### 3a. Use cross-validated log-loss for hyperparameter selection

**Current:** C is selected by maximizing training-set accuracy across 4 values.

**Paper:** Lambda (= 1/C) is chosen by **10-fold cross-validation** minimizing **deviance** (log-loss) within the training set, using R's `cv.glmnet`.

**Fix:** Use `GridSearchCV` with `scoring='neg_log_loss'` and `cv=10`. Expand the C grid (e.g., `np.logspace(-4, 2, 20)`). Or use `LogisticRegressionCV` with `scoring='neg_log_loss'`.

**File:** Regression scripts.

### 3b. Use proper multinomial logistic regression

**Current:** `SGDClassifier` with `loss='log_loss'` uses one-vs-rest by default, not multinomial.

**Paper:** Multinomial logistic regression (R `glmnet` with `family="multinomial"`).

**Fix:** Use `LogisticRegression` with `solver='saga'` (supports elastic net and multinomial). Note: sklearn 1.8+ deprecated the `penalty` parameter; use `l1_ratio=0.5` and set `C` directly. If memory is an issue, process in smaller batches or use `LogisticRegressionCV`.

**File:** Regression scripts.

### 3c. SUE.txt should use multinomial log-odds, not one-vs-rest

**Current:** Computes log-odds as `log(P(H)/(1-P(H))) - log(P(L)/(1-P(L)))` from `predict_proba` output. With SGDClassifier's OvR, these are not true multinomial log-odds.

**Paper (Eq. 1):** `SUE.txt = log-odds(H) - log-odds(L)` where log-odds come from the multinomial logistic model's linear predictor (`type="link"` in R).

**Fix:** Use a multinomial model and extract decision function values directly, or use `predict_proba` from a properly fitted multinomial model.

**File:** Regression scripts.

## Priority 4: Data Processing

### 4a. Winsorize continuous variables at 1% and 99%

**Current:** No winsorization.

**Paper (Online Appendix A):** All continuous variables are winsorized at the 1% and 99% levels.

**Fix:** After computing abnormal returns and CARs, winsorize at the 1st and 99th percentiles.

**File:** Dataset construction scripts.

### 4b. Panel regression with fixed effects and clustered SE

**Current:** Pooled OLS with intercept and standardized SUE.txt.

**Paper (Table 5):** CAR regressed on SUE.txt with **firm fixed effects** and **year-quarter fixed effects**, standard errors **clustered by firm and year-quarter**.

**Fix:** Use `linearmodels.PanelOLS` or demean manually. Implement two-way clustering using the Cameron-Gelbach-Miller method or `linearmodels`.

**File:** Regression result reporting section.

### 4c. Compustat matching needs causal direction and tolerance

**Current:** Matches each transcript to the nearest Compustat record by absolute date difference.

**Paper:** Each earnings call should match to the quarterly record whose `rdq` (report date) is closest and within a reasonable window (e.g., within 5 days), with priority to records where the call date is on or after `rdq`.

**Fix:** Filter to Compustat records where `rdq <= call_date + 5 days` and `rdq >= call_date - 5 days`, then pick the closest.

**File:** `build_dataset_lean.py:266-275`.

### 4d. Body text extraction from XML

**Current:** `body_el.text` only reads direct text content.

**Fix:** Use `"".join(body_el.itertext())` to capture text from any nested elements.

**File:** `parse_transcripts.py:49`.

## Implementation Order

1. Fix `#` token pattern (2c) -- trivial, immediate impact
2. Fix CRSP sentinel codes (1c) -- trivial
3. Fix winsorization (4a) -- trivial
4. Fix XML body extraction (4d) -- trivial
5. Add Q&A text and bigrams (2a) -- moderate, requires memory management
6. Switch to benchmark-adjusted CARs (1a) -- moderate, requires FF6 portfolio data
7. Implement rolling vocabulary (2b) -- moderate, restructures feature pipeline
8. Fix hyperparameter CV (3a) -- moderate
9. Switch to multinomial logistic (3b, 3c) -- moderate
10. Implement event-day timing heuristic (1b) -- moderate, requires XML parsing changes
11. Add panel FE regression (4b) -- moderate, may need `linearmodels` package
12. Fix Compustat matching (4c) -- low priority
