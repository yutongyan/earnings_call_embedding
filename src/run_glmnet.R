#!/usr/bin/env Rscript
# Exact PEAD.txt replication using cv.glmnet multinomial elastic net.
#
# Reads pre-computed features (Matrix Market) from data/glmnet_windows/{quarter}/
# Runs cv.glmnet(family="multinomial", alpha=0.5, nfolds=10)
# Computes SUE.txt = link_H - link_L via predict(type="link")
#
# Usage:
#   Rscript src/run_glmnet.R

library(glmnet)
library(Matrix)

data_dir <- "data/glmnet_windows"
output_file <- "data/sue_txt_glmnet.csv"

# Get all quarter directories
quarters <- sort(list.dirs(data_dir, recursive = FALSE, full.names = FALSE))
cat(sprintf("Found %d quarters\n", length(quarters)))

all_results <- data.frame()

for (i in seq_along(quarters)) {
    q <- quarters[i]
    q_dir <- file.path(data_dir, q)

    # Check required files
    if (!file.exists(file.path(q_dir, "X_train.mtx"))) {
        cat(sprintf("  %s: missing files, skipping\n", q))
        next
    }

    # Read sparse matrices
    X_train <- readMM(file.path(q_dir, "X_train.mtx"))
    X_test <- readMM(file.path(q_dir, "X_test.mtx"))

    # Convert to dgCMatrix (compressed sparse column) for glmnet
    X_train <- as(X_train, "dgCMatrix")
    X_test <- as(X_test, "dgCMatrix")

    # Read labels
    y_train <- scan(file.path(q_dir, "y_train.csv"), what = character(), quiet = TRUE)
    y_train <- factor(y_train, levels = c("H", "F", "L"))

    # Read test metadata
    meta_test <- read.csv(file.path(q_dir, "meta_test.csv"), stringsAsFactors = FALSE)

    # Skip if too few observations or missing classes
    if (nrow(X_train) < 100 || length(unique(y_train)) < 3) {
        cat(sprintf("  %s: insufficient data, skipping\n", q))
        next
    }

    # Fit cv.glmnet: multinomial elastic net, alpha=0.5, 10-fold CV
    set.seed(42)
    tryCatch({
        fit <- cv.glmnet(
            x = X_train,
            y = y_train,
            family = "multinomial",
            alpha = 0.5,
            nfolds = 10,
            type.measure = "deviance",
            standardize = TRUE
        )

        # Predict log-odds (type="link") on test set
        preds <- predict(fit, newx = X_test, s = "lambda.min", type = "link")

        # predict(type="link") returns 3D array: [n_test, n_class, n_lambda]
        # dim = [2362, 3, 1], dimnames = [NULL, c("H","F","L"), "lambda.min"]
        link_H <- preds[, "H", 1]
        link_L <- preds[, "L", 1]

        # SUE.txt = log-odds(H) - log-odds(L)
        sue_txt <- link_H - link_L

        # Build results
        q_results <- data.frame(
            event_id = meta_test$event_id,
            permno = meta_test$permno,
            call_date = meta_test$call_date,
            yq = q,
            sue_txt = sue_txt,
            abnormal_return = meta_test$abnormal_return,
            lambda_min = fit$lambda.min,
            stringsAsFactors = FALSE
        )

        all_results <- rbind(all_results, q_results)

        cat(sprintf("[%d/%d] %s: n_test=%d, lambda=%.6f, SUE.txt mean=%.3f std=%.3f\n",
                    i, length(quarters), q, nrow(X_test),
                    fit$lambda.min, mean(sue_txt), sd(sue_txt)))

    }, error = function(e) {
        cat(sprintf("  %s: ERROR %s\n", q, e$message))
    })

    # Clean up
    rm(X_train, X_test, y_train)
    gc()
}

# Save results
write.csv(all_results, output_file, row.names = FALSE)
cat(sprintf("\nSaved %d results to %s\n", nrow(all_results), output_file))

# Print summary
cat("\n=== SUE.txt Distribution ===\n")
cat(sprintf("  mean: %.4f\n", mean(all_results$sue_txt)))
cat(sprintf("  std:  %.4f\n", sd(all_results$sue_txt)))
cat(sprintf("  min:  %.4f\n", min(all_results$sue_txt)))
cat(sprintf("  max:  %.4f\n", max(all_results$sue_txt)))
