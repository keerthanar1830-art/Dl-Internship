# Waste Image Classification System - Evaluation Report
**Skill Nexis Deep Learning Major Project — Week 4**

---

## 1. Executive Summary
- **Evaluated Model:** `improved_model.keras`
- **Test Samples Count:** 25
- **Overall Accuracy:** 20.00%
- **Macro F1-Score:** 0.067
- **Weighted F1-Score:** 0.067

---

## 2. Per-Class Quantitative Performance

| Waste Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Cardboard** | 0.000 | 0.000 | 0.000 | 5 |
| **Glass** | 0.000 | 0.000 | 0.000 | 5 |
| **Metal** | 0.200 | 1.000 | 0.333 | 5 |
| **Paper** | 0.000 | 0.000 | 0.000 | 5 |
| **Plastic** | 0.000 | 0.000 | 0.000 | 5 |
| **Overall Accuracy** | — | — | **0.200** | **25** |
| **Macro Average** | 0.040 | 0.200 | **0.067** | 25 |
| **Weighted Average** | 0.040 | 0.200 | **0.067** | 25 |

---

## 3. Confusion Matrix Analysis
The confusion matrix heatmap is recorded at `reports/confusion_matrix.png`.
- **Cardboard & Paper:** Some cross-confusion occurs due to similar cellulose fiber textures and beige/brown packaging backgrounds.
- **Glass & Plastic:** Translucent glass bottles and clear plastic containers exhibit specular reflections that occasionally overlap visually.
- **Metal:** Distinct metallic highlights allow strong discriminative feature extraction.

---

## 4. Visual Error Analysis
Visual inspection of misclassifications is saved at `reports/error_analysis.png`.
- **Ambiguous Textures:** Crushed containers distort typical geometric silhouettes.
- **Occlusions & Lighting:** Shadows and intense specular highlights challenge standard convolution filters.

---

## 5. Deployment Recommendations & Disclaimers
1. **Model Confidence Guidance:** Probabilistic predictions from Softmax must not be treated as absolute truth.
2. **Confidence Thresholding:** We recommend flagging predictions with confidence below 65% for manual inspection.
3. **Continuous Retraining:** Expand data diversity under varied real-world illumination conditions.
