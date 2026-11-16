# Solution 06 — Learning resources

## Data science foundations

* Kaggle Learn (free, hands-on): Pandas, Intro to ML, Intermediate ML, Feature Engineering: <https://www.kaggle.com/learn>
* scikit-learn user guide, "Outlier and novelty detection" (Isolation Forest): <https://scikit-learn.org/stable/modules/outlier_detection.html>
* XGBoost documentation, tutorials and parameter tuning: <https://xgboost.readthedocs.io/>

## Fraud detection specifically

* "Reproducible Machine Learning for Credit Card Fraud Detection: Practical Handbook" (free online book, ULB):
  <https://fraud-detection-handbook.github.io/fraud-detection-handbook/>
* Imbalanced data and precision-recall curves: scikit-learn "Precision-Recall" example
  <https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html>
* Explainability with SHAP values: <https://shap.readthedocs.io/> (XGBoost's `pred_contribs` returns the same values)

## Time-series anomaly detection (extension)

* Granite TimeSeries TSPulse model card and notebooks: <https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1>
* granite-tsfm repository: <https://github.com/ibm-granite/granite-tsfm>

## LLM explanations

* Granite 4.2 model card: <https://huggingface.co/ibm-granite/granite-4.2-8b>

## IBM Power

* ppc64le/pyeco XGBoost example (the IBM wheel this solution uses): <https://github.com/ppc64le/pyeco/tree/main/examples>
* IBM Power wheel index: <https://wheels.developerfirst.ibm.com/ppc64le/linux>
* RocketCE conda channel (older xgboost 2.0.3 builds): <https://anaconda.org/rocketce>
* IBM Power Virtual Server docs: <https://cloud.ibm.com/docs/power-iaas>

## Practice exercises

1. Add a feature: number of distinct governorates the customer used in the last 7 days.
2. Plot the precision-recall curve and pick `alert_rate` from it.
3. Make `retrain_with_feedback.py` compare old and new models on the **same** latest test window.
4. Export the XGBoost model to ONNX and score it with onnxruntime (IBM wheel); compare latency.
