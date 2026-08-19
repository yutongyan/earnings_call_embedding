# PEAD.txt: Post-Earnings-Announcement Drift Using Text

**Authors:** Vitaly Meursault, Pierre Jinghong Liang, Bryan R. Routledge, Madeline Marco Scanlon

**Journal:** Journal of Financial and Quantitative Analysis, Vol. 58, Issue 6, pp. 2299-2326 (2022)

**DOI:** 10.1017/S0022109022001181

**Keywords:** PEAD, Machine Learning, NLP, Text Analysis

**JEL:** G14, G12, C00

## Research Question

Can the text of earnings conference calls, independent of the reported earnings number, predict post-earnings-announcement drift? The paper proposes a purely text-based measure of earnings surprises and tests whether it generates drift comparable to (or larger than) the classic numerical PEAD.

## Key Contribution

The authors construct **SUE.txt** (Standardized Unexpected Earnings text), a new numerical measure of earnings announcement surprises derived entirely from the text of earnings calls without explicitly incorporating the reported earnings value. SUE.txt generates a text-based post-earnings-announcement drift (**PEAD.txt**) that is larger than the classic PEAD across the full sample. PEAD.txt remains substantial in recent years when the classic PEAD has shrunk close to zero.

## Methodology

### Predictive Model

- A **regularized logistic text regression** (elastic net) classifies one-day abnormal returns around earnings calls into three categories: *high*, *flat*, and *low*.
- Features: log frequencies of the 1,000 most common **unigrams** and 1,000 most common **bigrams** from both the management presentation and Q&A sections separately (4,000 variables total).
- Numbers in transcripts are replaced with a generic token "#" (e.g., "$1000.00" becomes "$#"), so the model does not directly use reported earnings figures.
- The model is re-estimated every quarter using a **rolling 8-quarter training window** with 10-fold cross-validation, ensuring strictly out-of-sample predictions.

### SUE.txt Construction

SUE.txt is defined as the difference in log-odds between the high-return and low-return categories:

> SUE.txt = log-odds(H) - log-odds(L)

Positive values indicate good news; negative values indicate bad news; zero means no unexpected information.

### Abnormal Returns

- Announcement-day abnormal returns are computed using the **Fama-French three-factor plus momentum model** via the WRDS Event Studies tool.
- Long-run cumulative abnormal returns (CAR) use the **Fama-French six size and book-to-market matched portfolios** as benchmarks.

## Data

- **Source:** Capital IQ Transcripts database via WRDS, merged with CRSP, Compustat, and IBES.
- **Sample:** 108,704 earnings call transcripts from 2008Q1 to 2019Q4; final dataset of 85,160 observations (2010Q1-2019Q4) covering 4,701 unique firms.
- **Document size:** Median presentation section is ~3,000 words; median Q&A section is ~4,000 words.

## Main Results

### PEAD.txt vs. Classic PEAD

Using quintile spread portfolios (long top quintile, short bottom quintile) over a 252-trading-day horizon:

| Horizon (trading days) | PEAD.txt | Classic PEAD |
|------------------------|----------|--------------|
| 63 (1 quarter)         | 2.87%    | 1.54%        |
| 126 (2 quarters)       | 4.61%    | 2.70%        |
| 189 (3 quarters)       | 6.51%    | 3.87%        |
| 252 (1 year)           | 8.01%    | 4.63%        |

PEAD.txt is larger at every horizon, and the gap widens over time.

### Panel Regressions

- The association between SUE.txt and abnormal returns is **more than twice as strong** as between classic SUE and abnormal returns.
- Results are robust to firm and year-quarter fixed effects, clustering by firm and year-quarter.
- A one standard deviation increase in SUE.txt is associated with 3% to 6% of a standard deviation increase in 63-day CAR.

### Trading Strategy Alpha

- A spread portfolio based on SUE.txt quintiles earns a daily alpha of **3.9 basis points** (statistically significant) under the Fama-French five factors plus momentum.
- The classic SUE spread portfolio alpha is lower at **2.6 basis points**.
- An equally-weighted combination of SUE.txt and SUE (SUE.mix) produces the best overall alpha of **4.2 basis points**.

### Persistence Over Time

- PEAD.txt is larger than PEAD in eight out of ten years (exceptions: 2012, 2013).
- PEAD.txt never falls below 3.4% at the calendar-year mark, even as classic PEAD approaches zero in some years.
- PEAD.txt shows a resurgence in 2019, suggesting it is more robust to forces that have been reducing classic PEAD.

## Text and Numbers Are Complementary

- Text and numbers together classify announcement-day returns better than either alone.
- However, the text-only model produces the **largest drift** (8.01%), versus the Text+Num model (6.15%) and the Num-only model (4.11%).
- The authors speculate that numerical information is incorporated more quickly by markets, while textual information takes longer, making pure text surprises most associated with the drift.
- Rank-aggregating SUE.txt and SUE into **SUE.mix** produces the largest overall drift of 8.87%.

## Analytic Tools for Explaining PEAD.txt

### Word-Level Impact

- Word impact is defined as the product of the model coefficient and the mean log frequency: I_j = (beta_H - beta_L) * mean(x_j).
- High positive impact tokens: "favorable," "strong," "improvement," "nice," "good."
- High negative impact tokens: "issue," "loss," "decline," "lower," "impacted."
- Words can be impactful in two ways: rare words with large coefficients (e.g., "nice") or common words with moderate coefficients (e.g., "good," "not").

### Paragraph-Level SUE.txt

- Document-level SUE.txt decomposes into paragraph-level contributions (**SUE.txt^P**).
- Paragraphs are classified into five groups based on business curriculum keywords:
  1. **Financial accounting:** bottom line, metrics, adjustments, lending, financing
  2. **Operations management and marketing:** operational metrics, segments, supply chain, production, interruptions, marketing
  3. **Global economics:** foreign exchange, seasonality/weather, general global economics
  4. **Strategy:** competition, expansion, contraction, partners, deals, government, restructuring, general strategy
  5. **Forward-looking:** paragraphs containing forward-looking phrases

### Cross-Section of Paragraph Content

- Bottom line, forex, interruptions, and seasonality paragraphs have the highest mean absolute SUE.txt^P (most surprising per paragraph).
- Financial accounting metrics paragraphs dominate overall contribution to SUE.txt due to their high prevalence (~37% of paragraphs).
- Rare topics (operational interruptions, foreign exchange) drive extreme surprise values when they appear.

## Autocorrelation of SUE.txt

- SUE.txt exhibits positive autocorrelation, similar to classic SUE.
- Unlike SUE, SUE.txt is **more persistent when earnings volatility is higher**.
- SUE.txt is more mean-reverting for loss firms, consistent with findings for classic SUE.

## Why a Linear Model?

The authors choose regularized logistic regression over deep learning (e.g., BERT) for three reasons:

1. **Interpretability:** Coefficients directly indicate which words drive predictions.
2. **Computational efficiency:** 4,000 parameters vs. millions for BERT/deep learning.
3. **Long documents:** Earnings calls are thousands of words; BERT's 512-token limit would require lossy partitioning.

The authors acknowledge that deep learning models could improve predictive accuracy by incorporating word context, but at the cost of interpretability and computational tractability.

## Conclusion

SUE.txt flexibly summarizes good and bad news about firms from earnings call text, parallel to how numerical earnings summarize the same underlying economic activity. The text-based drift (PEAD.txt) is larger than classical PEAD and persists even as classic PEAD diminishes, deepening the PEAD puzzle. The results suggest that investor underreaction to earnings announcements extends far beyond the headline earnings number to encompass a wide range of qualitative information discussed in conference calls.

## Relevance to This Project

This paper provides the foundational motivation for using earnings call text embeddings to capture market-relevant information. Key takeaways for our work:

- Earnings call transcripts contain rich, return-predictive information beyond reported earnings numbers.
- Simple bag-of-words features already generate significant drift; modern embeddings (e.g., transformer-based) could potentially capture even more.
- The text-based surprise measure complements rather than substitutes for numerical earnings surprises.
- The rolling-window estimation approach ensures out-of-sample validity and avoids look-ahead bias.
