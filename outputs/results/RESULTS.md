# HandTap2PDScore - evaluation results

HUBU-FIS finger tapping, 234 videos / 118 participants. Metrics are weighted averages (as in UBU-PD-FT-Assessment). Subject CV = 5 x 5-fold StratifiedGroupKFold by participant, mean ± SD over repeats; LOO = leave-one-video-out (reference protocol; optimistic because the other hand of the same person is in training).


## MDS-UPDRS 3.4 score (0-3)


### subject CV - best model per feature set (selected by MCC)

| Feature set | Model | MCC | F1 | Accuracy | A_AC | Precision | Recall |
|---|---|---|---|---|---|---|---|
| kin_rtmpose | ordinal_rf | 0.515 ± 0.015 | 0.671 ± 0.011 | 0.668 ± 0.010 | 0.996 ± 0.000 | 0.692 ± 0.009 | 0.668 ± 0.010 |
| kin_fusion | svm | 0.493 ± 0.033 | 0.652 ± 0.022 | 0.656 ± 0.021 | 0.973 ± 0.003 | 0.654 ± 0.020 | 0.656 ± 0.021 |
| kin_mediapipe | logreg | 0.474 ± 0.025 | 0.635 ± 0.019 | 0.639 ± 0.021 | 0.964 ± 0.011 | 0.637 ± 0.019 | 0.639 ± 0.021 |
| ref_mediapipe | rf | 0.463 ± 0.021 | 0.626 ± 0.014 | 0.637 ± 0.013 | 0.957 ± 0.010 | 0.626 ± 0.011 | 0.637 ± 0.013 |
| ref_rtmpose | ordinal_rf | 0.440 ± 0.014 | 0.620 ± 0.008 | 0.611 ± 0.009 | 0.985 ± 0.002 | 0.664 ± 0.013 | 0.611 ± 0.009 |
| kin_vitpose | xgb | 0.364 ± 0.035 | 0.564 ± 0.024 | 0.573 ± 0.023 | 0.950 ± 0.014 | 0.566 ± 0.024 | 0.573 ± 0.023 |

### loo CV - best model per feature set (selected by MCC)

| Feature set | Model | MCC | F1 | Accuracy | A_AC | Precision | Recall |
|---|---|---|---|---|---|---|---|
| kin_rtmpose | ordinal_rf | 0.552 | 0.697 | 0.692 | 0.996 | 0.713 | 0.692 |
| kin_mediapipe | svm | 0.529 | 0.676 | 0.675 | 0.966 | 0.676 | 0.675 |
| kin_fusion | rf | 0.508 | 0.658 | 0.662 | 0.974 | 0.656 | 0.662 |
| ref_mediapipe | rf | 0.499 | 0.653 | 0.658 | 0.970 | 0.650 | 0.658 |
| ref_rtmpose | xgb | 0.491 | 0.639 | 0.658 | 0.957 | 0.651 | 0.658 |
| kin_vitpose | ordinal_rf | 0.365 | 0.567 | 0.564 | 0.987 | 0.616 | 0.564 |

### All models, subject CV (MCC)

| featureset    |   knn |   logreg |   ordinal_rf |    rf |   svm |   xgb |
|:--------------|------:|---------:|-------------:|------:|------:|------:|
| kin_fusion    | 0.381 |    0.477 |        0.472 | 0.493 | 0.493 | 0.46  |
| kin_mediapipe | 0.349 |    0.474 |        0.453 | 0.469 | 0.462 | 0.471 |
| kin_rtmpose   | 0.394 |    0.423 |        0.515 | 0.489 | 0.487 | 0.452 |
| kin_vitpose   | 0.233 |    0.265 |        0.343 | 0.352 | 0.287 | 0.364 |
| ref_mediapipe | 0.367 |    0.42  |        0.458 | 0.463 | 0.398 | 0.443 |
| ref_rtmpose   | 0.32  |    0.396 |        0.44  | 0.415 | 0.367 | 0.42  |

## PD vs control


### subject CV - best model per feature set (selected by MCC)

| Feature set | Model | MCC | F1 | Accuracy | Precision | Recall | Sensitivity | Specificity | ROC_AUC |
|---|---|---|---|---|---|---|---|---|---|
| ref_rtmpose | logreg | 0.466 ± 0.051 | 0.745 ± 0.027 | 0.742 ± 0.027 | 0.755 ± 0.023 | 0.742 ± 0.027 | 0.749 ± 0.037 | 0.729 ± 0.031 | 0.798 ± 0.018 |
| kin_rtmpose | rf | 0.462 ± 0.041 | 0.750 ± 0.019 | 0.749 ± 0.018 | 0.751 ± 0.019 | 0.749 ± 0.018 | 0.791 ± 0.017 | 0.675 ± 0.040 | 0.806 ± 0.013 |
| ref_mediapipe | rf | 0.442 ± 0.022 | 0.740 ± 0.011 | 0.738 ± 0.012 | 0.742 ± 0.010 | 0.738 ± 0.012 | 0.779 ± 0.015 | 0.668 ± 0.009 | 0.782 ± 0.005 |
| kin_fusion | rf | 0.424 ± 0.032 | 0.731 ± 0.013 | 0.729 ± 0.012 | 0.734 ± 0.015 | 0.729 ± 0.012 | 0.768 ± 0.015 | 0.661 ± 0.043 | 0.794 ± 0.011 |
| kin_mediapipe | logreg | 0.381 ± 0.040 | 0.704 ± 0.018 | 0.700 ± 0.018 | 0.716 ± 0.019 | 0.700 ± 0.018 | 0.711 ± 0.024 | 0.680 ± 0.047 | 0.752 ± 0.016 |
| kin_vitpose | xgb | 0.343 ± 0.045 | 0.698 ± 0.021 | 0.703 ± 0.023 | 0.697 ± 0.022 | 0.703 ± 0.023 | 0.800 ± 0.034 | 0.532 ± 0.014 | 0.743 ± 0.028 |

### loo CV - best model per feature set (selected by MCC)

| Feature set | Model | MCC | F1 | Accuracy | Precision | Recall | Sensitivity | Specificity | ROC_AUC |
|---|---|---|---|---|---|---|---|---|---|
| kin_rtmpose | xgb | 0.549 | 0.788 | 0.786 | 0.792 | 0.786 | 0.805 | 0.753 | 0.827 |
| ref_rtmpose | logreg | 0.519 | 0.769 | 0.765 | 0.780 | 0.765 | 0.758 | 0.776 | 0.833 |
| kin_fusion | rf | 0.497 | 0.763 | 0.761 | 0.768 | 0.761 | 0.779 | 0.729 | 0.815 |
| ref_mediapipe | logreg | 0.484 | 0.748 | 0.744 | 0.766 | 0.744 | 0.725 | 0.776 | 0.801 |
| kin_mediapipe | svm | 0.468 | 0.736 | 0.731 | 0.760 | 0.731 | 0.698 | 0.788 | 0.782 |
| kin_vitpose | xgb | 0.410 | 0.729 | 0.735 | 0.729 | 0.735 | 0.839 | 0.553 | 0.795 |

### All models, subject CV (MCC)

| featureset    |   knn |   logreg |    rf |   svm |   xgb |
|:--------------|------:|---------:|------:|------:|------:|
| kin_fusion    | 0.279 |    0.386 | 0.424 | 0.418 | 0.413 |
| kin_mediapipe | 0.264 |    0.381 | 0.358 | 0.374 | 0.374 |
| kin_rtmpose   | 0.32  |    0.389 | 0.462 | 0.433 | 0.432 |
| kin_vitpose   | 0.207 |    0.251 | 0.307 | 0.289 | 0.343 |
| ref_mediapipe | 0.344 |    0.421 | 0.442 | 0.417 | 0.418 |
| ref_rtmpose   | 0.332 |    0.466 | 0.421 | 0.378 | 0.393 |

## PD vs control per participant (mean P(PD) of both hands, subject CV)

| Feature set | Model | MCC | F1 | Accuracy | Precision | Recall | Sensitivity | Specificity | ROC_AUC |
|---|---|---|---|---|---|---|---|---|---|
| ref_rtmpose | logreg | 0.503 ± 0.053 | 0.764 ± 0.027 | 0.761 ± 0.028 | 0.771 ± 0.024 | 0.761 ± 0.028 | 0.771 ± 0.036 | 0.744 ± 0.025 | 0.828 ± 0.019 |
| kin_fusion | xgb | 0.495 ± 0.025 | 0.767 ± 0.011 | 0.769 ± 0.012 | 0.767 ± 0.012 | 0.769 ± 0.012 | 0.845 ± 0.025 | 0.637 ± 0.019 | 0.819 ± 0.013 |
| kin_rtmpose | rf | 0.469 ± 0.045 | 0.753 ± 0.020 | 0.753 ± 0.020 | 0.754 ± 0.021 | 0.753 ± 0.020 | 0.797 ± 0.013 | 0.674 ± 0.042 | 0.838 ± 0.014 |
| kin_mediapipe | xgb | 0.433 ± 0.046 | 0.739 ± 0.021 | 0.744 ± 0.020 | 0.739 ± 0.021 | 0.744 ± 0.020 | 0.840 ± 0.015 | 0.577 ± 0.031 | 0.815 ± 0.015 |
| ref_mediapipe | rf | 0.431 ± 0.032 | 0.736 ± 0.016 | 0.736 ± 0.016 | 0.736 ± 0.015 | 0.736 ± 0.016 | 0.789 ± 0.023 | 0.642 ± 0.011 | 0.804 ± 0.005 |
| kin_vitpose | xgb | 0.399 ± 0.046 | 0.722 ± 0.021 | 0.732 ± 0.020 | 0.725 ± 0.022 | 0.732 ± 0.020 | 0.859 ± 0.016 | 0.512 ± 0.033 | 0.773 ± 0.030 |

## Feature associations (MediaPipe + RTMPose consensus features)

Spearman ρ with UPDRS, Mann-Whitney AUC PD vs control (AUC > 0.5 = higher in PD), BH-FDR q-values. Videos, not participants, are the unit (2 hands per person), so p-values are somewhat anti-conservative.

| feature             |   spearman_rho_updrs |   q_updrs |   auc_pd_vs_ctrl |   q_pd |   median_ctrl |   median_pd |
|:--------------------|---------------------:|----------:|-----------------:|-------:|--------------:|------------:|
| speed_mean          |              -0.7014 |    0      |           0.246  | 0      |        4.2801 |      3.0953 |
| close_speed_mean    |              -0.6617 |    0      |           0.2215 | 0      |       12.9675 |      9.2735 |
| open_speed_mean     |              -0.596  |    0      |           0.2223 | 0      |       10.3672 |      7.8248 |
| vel_peaks_per_tap   |               0.5755 |    0      |           0.7412 | 0      |        2.05   |      2.25   |
| amp_first5          |              -0.5639 |    0      |           0.2561 | 0      |        1.5468 |      1.2411 |
| speed_cv            |               0.5565 |    0      |           0.729  | 0      |        0.1064 |      0.1432 |
| amp_mean            |              -0.5537 |    0      |           0.2456 | 0      |        1.4727 |      1.1024 |
| amp_median          |              -0.5508 |    0      |           0.2473 | 0      |        1.469  |      1.1304 |
| peak_aperture_mean  |              -0.5329 |    0      |           0.2635 | 0      |        1.5953 |      1.239  |
| angle_amp_mean      |              -0.5295 |    0      |           0.2523 | 0      |       45.9916 |     33.4142 |
| amp_max             |              -0.5123 |    0      |           0.2757 | 0      |        1.6839 |      1.3789 |
| amp_cv              |               0.4942 |    0      |           0.7347 | 0      |        0.0741 |      0.1191 |
| iti_cv              |               0.4893 |    0      |           0.6045 | 0.0129 |        0.1092 |      0.1477 |
| halt_time_s         |               0.4066 |    0      |           0.5668 | 0.0124 |        0      |      0      |
| n_halts             |               0.406  |    0      |           0.5664 | 0.0123 |        0      |      0      |
| duty_cycle          |              -0.3942 |    0      |           0.3484 | 0.0003 |        0.6243 |      0.5904 |
| iti_mean_s          |               0.3922 |    0      |           0.604  | 0.0123 |        0.6369 |      0.7443 |
| tap_rate_hz         |              -0.3897 |    0      |           0.3958 | 0.0126 |        1.5612 |      1.3518 |
| n_taps              |              -0.3846 |    0      |           0.3922 | 0.0116 |       31      |     27      |
| hesitation_rate     |               0.3836 |    0      |           0.5892 | 0.0005 |        0      |      0      |
| n_hesitations       |               0.3832 |    0      |           0.5885 | 0.0005 |        0      |      0      |
| open_close_ratio    |              -0.3823 |    0      |           0.3539 | 0.0005 |        1.6667 |      1.4416 |
| dom_freq_hz         |              -0.3753 |    0      |           0.3983 | 0.0139 |        1.502  |      1.2517 |
| iti_median_s        |               0.3596 |    0      |           0.5956 | 0.0194 |        0.6323 |      0.7324 |
| iti_max_over_median |               0.357  |    0      |           0.5767 | 0.0577 |        1.2692 |      1.3094 |
| sparc               |              -0.3406 |    0      |           0.4192 | 0.0464 |       -7.9868 |     -8.5108 |
| spectral_peak_ratio |              -0.2988 |    0      |           0.4246 | 0.0605 |        0.8664 |      0.8472 |
| spectral_entropy    |               0.2491 |    0.0002 |           0.5701 | 0.077  |        0.4417 |      0.4659 |
| amp_slope_pct       |              -0.2298 |    0.0005 |           0.3745 | 0.003  |       -0.4279 |     -0.6659 |
| speed_slope_pct     |              -0.1678 |    0.0121 |           0.4015 | 0.0164 |       -0.6566 |     -0.8623 |
| min_aperture_mean   |               0.161  |    0.0159 |           0.6238 | 0.0033 |        0.092  |      0.111  |
| amp_decrement       |              -0.1342 |    0.0452 |           0.4008 | 0.0162 |       -0.0903 |     -0.1285 |
| wrist_rms_pl        |               0.1092 |    0.1042 |           0.5895 | 0.0284 |        0.15   |      0.167  |
| amp_last5_ratio     |              -0.1083 |    0.1041 |           0.4159 | 0.039  |        0.9011 |      0.8563 |
| iti_slope_pct       |               0.0743 |    0.265  |           0.5071 | 0.8566 |        0.1148 |      0.0463 |
| speed_decrement     |              -0.0732 |    0.2645 |           0.427  | 0.0673 |       -0.1389 |     -0.163  |
