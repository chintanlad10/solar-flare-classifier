### Logistic Regression and XGBosst Model summary
- **Logistic Regression (balanced)**: Standardized inputs with `StandardScaler`, `class_weight='balanced'`, `solver='lbfgs'`, and `max_iter=200`. Validation threshold tuning selected **0.500**. Best-threshold metrics: validation **TSS=0.929, HSS=0.928, ROC-AUC=0.992**; test **TSS=0.707, HSS=0.112, ROC-AUC=0.928**.
- **XGBoost (gradient boosted trees)**: `n_estimators=400`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`, and `scale_pos_weight=neg/pos`. Validation threshold tuning selected **0.550**. Best-threshold metrics: validation **TSS=0.964, HSS=0.963, ROC-AUC=0.999**; test **TSS=0.737, HSS=0.289, ROC-AUC=0.960**.
- **Takeaway**: XGBoost remains the stronger tabular baseline on the held-out test set, with better overall TSS/HSS and stronger precision-recall behavior under severe class imbalance.

### Results summary and interpretation

**Metric tables now use 3-decimal formatting. Values below match the latest executed outputs.**

| Model | Split | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | TSS | HSS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Validation (best threshold) | 0.500 | 0.964 | 0.954 | 0.974 | 0.964 | 0.992 | 0.990 | 0.929 | 0.928 |
| Logistic Regression | Test (best threshold) | 0.500 | 0.794 | 0.078 | 0.916 | 0.143 | 0.928 | 0.204 | 0.707 | 0.112 |
| XGBoost | Validation (best threshold) | 0.550 | 0.982 | 0.975 | 0.988 | 0.981 | 0.999 | 0.998 | 0.964 | 0.963 |
| XGBoost | Test (best threshold) | 0.550 | 0.933 | 0.193 | 0.802 | 0.311 | 0.960 | 0.370 | 0.737 | 0.289 |

- Validation performance is very strong for both models, with XGBoost leading on TSS/HSS and AUC metrics.
- On the imbalanced test split, Logistic Regression keeps high recall but low precision, which lowers HSS despite acceptable TSS.
- XGBoost provides the strongest overall tabular benchmark in this notebook and remains the better reference for sequence-model comparison.

---

### LSTM Whole Sequence Model (BCE) summary
- **Input space**: Uses the five selected attributes (`SAVNCPP`, `ABSNJZH`, `R_VALUE`, `EPSZ`, `TOTFZ`) with train-stat normalization and full-sequence modeling (no sliding windows).
- **Architecture & training**: 1-layer LSTM (`hidden=64`, `dropout=0.2`) with `BCEWithLogitsLoss(pos_weight=neg/pos)`, Adam (`1e-3`, weight decay `1e-4`), gradient clipping, and `ReduceLROnPlateau`; validation threshold sweep uses `0.1–0.9` with FPR constraint `<= 0.10`.
- **Latest run highlights**: Best validation threshold is **0.575** with validation **TSS=0.9223, HSS=0.9223, ROC-AUC=0.9919**; on test at the same threshold: **TSS=0.8002, HSS=0.1575, ROC-AUC=0.9651**.

### Results summary and interpretation

**Whole-sequence LSTM with BCE loss — latest executed run**

| Split / Setting | Threshold | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | TSS | HSS |
|-----------------|-----------|--------:|-------:|---------:|----------:|-------:|----:|----:|
| Validation @ best | 0.575 | 0.9919 | 0.9910 | 0.9612 | 0.9632 | 0.9586 | 0.9223 | 0.9223 |
| Test @ best (0.575) | 0.575 | 0.9651 | 0.3909 | 0.8419 | 0.1031 | 0.9605 | 0.8002 | 0.1575 |
| Test best sweep (FPR<=0.10) | 0.850 | 0.9651 | 0.3909 | 0.9063 | 0.1563 | 0.9041 | 0.8105 | 0.2422 |
| Validation @ 0.5 | 0.500 | 0.9919 | 0.9910 | 0.9603 | 0.9614 | 0.9587 | 0.9206 | 0.9206 |
| Test @ 0.5 | 0.500 | 0.9651 | 0.3909 | 0.8328 | 0.0985 | 0.9665 | 0.7967 | 0.1497 |

- Validation performance is strong with threshold=0.575 (TSS/HSS=0.9223) and low validation FPR.
- On the held-out test set, the model achieves very high recall (0.9605) but extremely low precision (0.1031), indicating a large number of false positives.
- This imbalance leads to a significant drop in HSS despite strong TSS, highlighting limitations of BCE under severe class imbalance.
- Applying an FPR-constrained threshold improves HSS (**0.1575 → 0.2422**) by reducing false positives, at the cost of some recall.
- Overall, BCE prioritizes recall but struggles to maintain precision, making it less reliable for rare-event detection.

---

### LSTM Whole Sequence Model (Focal Loss) summary
- **Input space**: Uses the five selected attributes (`SAVNCPP`, `ABSNJZH`, `R_VALUE`, `EPSZ`, `TOTFZ`) with train-stat normalization and full-sequence modeling (no sliding windows).
- **Architecture & training**: 1-layer LSTM (`hidden=64`, `dropout=0.2`) with **Focal Loss** (`alpha=0.75`, `gamma=2.0`) to address class imbalance, Adam (`1e-3`, weight decay `1e-4`), gradient clipping, and `ReduceLROnPlateau`; validation threshold sweep uses `0.1–0.9` with FPR constraint `<= 0.10`.
- **Latest run highlights**: Best validation threshold is **0.600** with validation **TSS=0.9126, HSS=0.9126, ROC-AUC=0.9912**; on test at the same threshold: **TSS=0.7863, HSS=0.2205, ROC-AUC=0.9608**.

### Results summary and interpretation

**Whole-sequence LSTM with Focal Loss — latest executed run**

| Split / Setting | Threshold | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | TSS | HSS |
|-----------------|-----------|--------:|-------:|---------:|----------:|-------:|----:|----:|
| Validation @ best | 0.600 | 0.9912 | 0.9892 | 0.9563 | 0.9550 | 0.9572 | 0.9126 | 0.9126 |
| Test @ best (0.600) | 0.600 | 0.9608 | 0.3668 | 0.8973 | 0.1426 | 0.8888 | 0.7863 | 0.2205 |
| Test best sweep (FPR<=0.10) | 0.625 | 0.9608 | 0.3668 | 0.9107 | 0.1573 | 0.8595 | 0.7712 | 0.2418 |
| Validation @ 0.5 | 0.500 | 0.9912 | 0.9892 | 0.9520 | 0.9424 | 0.9623 | 0.9041 | 0.9040 |

- Validation performance remains strong, with slightly lower TSS/HSS compared to BCE but still above 0.91.
- On the test set, Focal Loss significantly improves precision (**0.1031 → 0.1426**) and HSS (**0.1575 → 0.2205**) compared to BCE.
- This improvement comes from reducing false positives, as reflected in lower FPR and more balanced predictions.
- Under FPR constraint, the model achieves the best trade-off with **HSS=0.2418**, further improving reliability.
- Compared to BCE, Focal Loss sacrifices some recall but provides a better precision-recall balance, making it more suitable for highly imbalanced solar flare prediction.

---

### Sliding Window LSTM Model summary
- **Architecture**: Single-directional LSTM with 2 layers (`hidden=128`, `dropout=0.3`) and a linear head on the last hidden state. Training uses Adam (`lr=1e-3`, `weight_decay=1e-4`), gradient clipping, and `ReduceLROnPlateau`.
- **Windowing / sampling**: 32-step windows with stride 4, capped positive/negative windows per sample, tail-focused positive windows, negative subsampling, and sample-level max aggregation.
- **Evaluation setup**: Thresholds are swept in `0.20–0.90` under `FPR <= 0.05`, and evaluation is done using TSS, HSS, ROC-AUC, and PR-AUC on validation and held-out test data.

---

### Results summary — Sliding Window LSTM (BCE)

| Split / Setting                   | Threshold | Accuracy | Precision | Recall | F1     | ROC-AUC | PR-AUC | TSS    | HSS    |
|----------------------------------|-----------|---------:|----------:|-------:|-------:|--------:|-------:|-------:|-------:|
| Validation @ best threshold      | 0.725     | 0.9829   | 0.9904    | 0.9751 | 0.9826 | 0.9988  | 0.9989 | 0.9656 | 0.9657 |
| Test @ validation threshold      | 0.725     | 0.9241   | 0.1579    | 0.6999 | 0.2576 | 0.9372  | 0.2257 | 0.6282 | 0.2341 |
| Test best sweep (FPR-constrained)| 0.875     | 0.9465   | 0.1995    | 0.6115 | 0.3009 | 0.9372  | 0.2257 | 0.5644 | 0.2805 |
| Validation @ threshold=0.5       | 0.500     | 0.9815   | 0.9735    | 0.9897 | 0.9815 | 0.9988  | 0.9989 | 0.9630 | 0.9629 |
| Test @ threshold=0.5             | 0.500     | 0.8804   | 0.1177    | 0.8237 | 0.2059 | 0.9372  | 0.2257 | 0.7052 | 0.1789 |

---

### Results summary — Sliding Window LSTM (Focal Loss)

| Split / Setting                   | Threshold | Accuracy | Precision | Recall | F1     | ROC-AUC | PR-AUC | TSS    | HSS    |
|----------------------------------|-----------|---------:|----------:|-------:|-------:|--------:|-------:|-------:|-------:|
| Validation @ best threshold      | 0.625     | 0.9522   | 0.9495    | 0.9546 | 0.9520 | 0.9890  | 0.9873 | 0.9043 | 0.9043 |
| Test @ validation threshold      | 0.625     | 0.8475   | 0.1051    | 0.9450 | 0.1892 | 0.9578  | 0.3501 | 0.7907 | 0.1608 |
| Test best sweep (FPR-constrained)| 0.825     | 0.9523   | 0.2359    | 0.6861 | 0.3511 | 0.9578  | 0.3501 | 0.6435 | 0.3324 |
| Validation @ threshold=0.5       | 0.500     | 0.9461   | 0.9264    | 0.9687 | 0.9471 | 0.9890  | 0.9873 | 0.8925 | 0.8923 |
| Test @ threshold=0.5             | 0.500     | 0.8060   | 0.0868    | 0.9780 | 0.1595 | 0.9578  | 0.3501 | 0.7807 | 0.1294 |

---

### Bidirectional LSTM Whole Sequence Model summary
- **Input space**: Uses the five selected attributes (`SAVNCPP`, `ABSNJZH`, `R_VALUE`, `EPSZ`, `TOTFZ`) with train-stat normalization and full-sequence modeling (no sliding windows).
- **Architecture & training**: 1-layer Bidirectional LSTM (`hidden=64` per direction, `dropout=0.3`) with a linear head on concatenated forward and backward hidden states. Training uses Adam (`lr=1e-3`, `weight_decay=1e-4`), gradient clipping, and `ReduceLROnPlateau`.
- **Evaluation setup**: Threshold sweep in `0.1–0.9` with FPR constraint `<= 0.10`, using TSS, HSS, ROC-AUC, and PR-AUC for evaluation on validation and test splits.
- **Loss comparison**:
  - **BCE (pos_weight)**: Stable and slightly better under FPR-constrained settings.
  - **Focal Loss**: Slightly improves recall and TSS but increases false positives.

---

### Results summary — BiLSTM (BCE)

| Split / Setting                   | Threshold | ROC-AUC | PR-AUC | TPR    | FPR    | TSS    | HSS    |
|----------------------------------|-----------|--------:|-------:|-------:|-------:|-------:|-------:|
| Validation @ best threshold      | 0.65      | 0.9939  | 0.9926 | 0.9579 | 0.0338 | 0.9241 | 0.9241 |
| Test @ validation threshold      | 0.65      | 0.9636  | 0.3882 | 0.9551 | 0.1414 | 0.8137 | 0.1771 |
| Test best sweep (FPR-constrained)| 0.85      | 0.9636  | 0.3882 | 0.8911 | 0.0861 | 0.8049 | 0.2556 |

---

### Results summary — BiLSTM (Focal Loss)

| Split / Setting                   | Threshold | ROC-AUC | PR-AUC | TPR    | FPR    | TSS    | HSS    |
|----------------------------------|-----------|--------:|-------:|-------:|-------:|-------:|-------:|
| Validation @ best threshold      | 0.6000    | 0.9934  | 0.9926 | 0.9605 | 0.0440 | 0.9165 | 0.9165 |
| Test @ best threshold            | 0.6000    | 0.9627  | 0.3863 | 0.9641 | 0.1474 | 0.8167 | 0.1720 |
| Test best sweep (FPR-constrained)| 0.7000    | 0.9627  | 0.3863 | 0.8999 | 0.0953 | 0.8046 | 0.2375 |

---

### Decomposition LSTM Whole Sequence Model summary
- **Input space**: Uses the five selected attributes (`SAVNCPP`, `ABSNJZH`, `R_VALUE`, `EPSZ`, `TOTFZ`) with train-stat normalization and full-sequence modeling (no sliding windows).
- **Architecture & training**: A depthwise 1D convolution extracts a smoothed per-feature **trend** component, and the **residual** (original − trend) is concatenated with the trend and passed to a 1-layer LSTM (`hidden=64`, `dropout=0.3`) followed by a linear head. Training uses Adam (`lr=1e-3`, `weight_decay=1e-4`), gradient clipping, and `ReduceLROnPlateau`.
- **Evaluation setup**: Threshold sweep in `0.1–0.9` with FPR constraint `<= 0.10`, using TSS, HSS, ROC-AUC, and PR-AUC on validation and test splits.
- **Loss comparison**:
  - **BCE (pos_weight)**: More stable and slightly better precision under validation threshold.
  - **Focal Loss**: Improves performance under FPR-constrained settings with better TSS/HSS.

---

### Results summary — Decomposition LSTM (BCE)

| Split / Setting                   | Threshold | ROC-AUC | PR-AUC | TPR    | FPR    | TSS    | HSS    |
|----------------------------------|-----------|--------:|-------:|-------:|-------:|-------:|-------:|
| Validation @ best threshold      | 0.65      | 0.9924  | 0.9917 | 0.9635 | 0.0417 | 0.9218 | 0.9218 |
| Test @ best threshold            | 0.65      | 0.9634  | 0.4192 | 0.9211 | 0.1164 | 0.8047 | 0.2044 |
| Test best sweep (FPR-constrained)| 0.80      | 0.9634  | 0.4192 | 0.8781 | 0.0938 | 0.7843 | 0.2350 |

---

### Results summary — Decomposition LSTM (Focal Loss)

| Split / Setting                   | Threshold | ROC-AUC | PR-AUC | TPR    | FPR    | TSS    | HSS    |
|----------------------------------|-----------|--------:|-------:|-------:|-------:|-------:|-------:|
| Validation @ best threshold      | 0.7000    | 0.9920  | 0.9913 | 0.9617 | 0.0394 | 0.9223 | 0.9223 |
| Test @ best threshold            | 0.7000    | 0.9668  | 0.4031 | 0.9546 | 0.1413 | 0.8133 | 0.1772 |
| Test best sweep (FPR-constrained)| 0.8000    | 0.9668  | 0.4031 | 0.8819 | 0.0751 | 0.8068 | 0.2818 |

---

### Sliding Window Decomposition LSTM Model summary
- **Input space & windows**: Uses the five selected attributes (`SAVNCPP`, `ABSNJZH`, `R_VALUE`, `EPSZ`, `TOTFZ`) with train-stat normalization. 32-step windows (stride 4) are generated per sample, with tail-focused positive selection, negative subsampling, and controlled class balance, followed by sample-level aggregation.
- **Architecture & training**: Each window is decomposed into a smoothed **trend** (depthwise Conv1d) and **residual** (original − trend), concatenated and passed through a 2-layer LSTM (`hidden=128`, `dropout=0.3`) with a linear output head. Training uses Adam (`lr=1e-3`, `weight_decay=1e-4`), gradient clipping, and `ReduceLROnPlateau`.
- **Evaluation setup**: Window-level predictions are aggregated to sample-level. Thresholds are swept in `0.2–0.9` with `FPR <= 0.05`, and evaluated using TSS, HSS, ROC-AUC, and PR-AUC.

---

### Results summary — Sliding Window Decomposition LSTM (BCE)

| Split / Setting                   | Threshold | ROC-AUC | PR-AUC | TPR    | FPR    | TSS    | HSS    |
|----------------------------------|-----------|--------:|-------:|-------:|-------:|-------:|-------:|
| Validation @ best threshold      | 0.35      | 0.9878  | 0.9876 | 0.9521 | 0.0461 | 0.9060 | 0.9060 |
| Test @ best threshold            | 0.35      | 0.9380  | 0.3289 | 0.8722 | 0.1564 | 0.7158 | 0.1450 |
| Test sweep (FPR-constrained) best| 0.95      | 0.9380  | 0.3289 | 0.5861 | 0.0485 | 0.5376 | 0.2639 |

---

### Results summary — Sliding Window Decomposition LSTM (Focal Loss)

| Split / Setting                   | Threshold | ROC-AUC | PR-AUC | TPR    | FPR    | TSS    | HSS    |
|----------------------------------|-----------|--------:|-------:|-------:|-------:|-------:|-------:|
| Validation @ best threshold      | 0.7000    | 0.9911  | 0.9904 | 0.9423 | 0.0335 | 0.9088 | 0.9089 |
| Test @ best threshold            | 0.7000    | 0.9657  | 0.3949 | 0.9535 | 0.1269 | 0.8266 | 0.1959 |
| Test sweep (FPR-constrained) best| 0.8000    | 0.9657  | 0.3949 | 0.7421 | 0.0441 | 0.6980 | 0.3489 |

---

### TCN Whole Sequence Model summary
- **Input space**: Uses the five selected attributes (`SAVNCPP`, `ABSNJZH`, `R_VALUE`, `EPSZ`, `TOTFZ`) with train-stat normalization and full-sequence modeling.
- **Architecture & training**: A Temporal Convolutional Network (TCN) with dilations `[1, 2, 4, 8]`, residual connections, and global average pooling followed by a linear classification head. Training uses Adam (`lr=1e-3`, `weight_decay=1e-4`), gradient clipping, and `ReduceLROnPlateau`.
- **Evaluation setup**: Threshold sweep in `0.1–0.9` under `FPR <= 0.10`, evaluated using TSS, HSS, ROC-AUC, and PR-AUC.
- **Loss comparison**:
  - **BCE (pos_weight)**: Strong validation performance but weaker generalization on test.
  - **Focal Loss**: Improves test performance (higher TSS/HSS) by better handling class imbalance.

---

### Results summary — TCN (BCE)

| Split / Setting                   | Threshold | ROC-AUC | PR-AUC | TPR    | FPR    | TSS    | HSS    |
|----------------------------------|-----------|--------:|-------:|-------:|-------:|-------:|-------:|
| Validation @ best threshold      | 0.1500    | 0.9911  | 0.9904 | 0.9543 | 0.0364 | 0.9179 | 0.9180 |
| Test @ validation threshold      | 0.1500    | 0.9000  | 0.3158 | 0.6866 | 0.0763 | 0.6102 | 0.2182 |
| Test best sweep (FPR-constrained)| 0.1000    | 0.9000  | 0.3158 | 0.7007 | 0.0839 | 0.6168 | 0.2058 |

---

### Results summary — TCN (Focal Loss)

| Split / Setting                   | Threshold | ROC-AUC | PR-AUC | TPR    | FPR    | TSS    | HSS    |
|----------------------------------|-----------|--------:|-------:|-------:|-------:|-------:|-------:|
| Validation @ best threshold      | 0.4750    | 0.9834  | 0.9840 | 0.9374 | 0.0354 | 0.9020 | 0.9021 |
| Test @ validation threshold      | 0.4750    | 0.9435  | 0.3426 | 0.7708 | 0.0837 | 0.6871 | 0.2271 |
| Test best sweep (FPR-constrained)| 0.4250    | 0.9435  | 0.3426 | 0.7979 | 0.0954 | 0.7024 | 0.2103 |

---

### Soft Voting Ensemble Model summary
- **Input space**: Combines predictions from three models trained on the same five selected attributes (`SAVNCPP`, `ABSNJZH`, `R_VALUE`, `EPSZ`, `TOTFZ`), ensuring consistent data alignment and normalization across all models.
- **Models combined**:
  - Sliding Window + Decomposition LSTM (Focal Loss)
  - Decomposition LSTM (Focal Loss)
  - XGBoost (tabular baseline)
- **Ensemble method**: Soft voting (simple probability averaging), where final prediction probability is computed as the mean of individual model probabilities.
- **Evaluation setup**: Threshold sweep in `0.1–0.9` under `FPR <= 0.10`, evaluated using TSS, HSS, ROC-AUC, and PR-AUC.

---

### Results summary — Soft Voting Ensemble

| Split / Setting                   | Threshold | ROC-AUC | PR-AUC | TPR    | FPR    | TSS    | HSS    |
|----------------------------------|-----------|--------:|-------:|-------:|-------:|-------:|-------:|
| Test best sweep (FPR-constrained)| 0.525     | 0.9624  | 0.4106 | 0.8733 | 0.0943 | 0.7790 | 0.2326 |

---

### Interpretation

- The ensemble improves prediction stability by combining outputs from diverse models, including sequence-based and tabular approaches.
- It achieves balanced performance with controlled FPR under constraint, demonstrating robustness in handling imbalanced data.
- However, the ensemble does not outperform the best individual model (Sliding Window + Decomposition LSTM with Focal Loss), indicating that averaging may dilute strong model-specific signals.
- The ensemble is therefore useful for improving reliability but is not selected as the final best-performing model.