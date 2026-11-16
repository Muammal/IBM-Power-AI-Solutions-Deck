# Solution 05 — Learning resources

## Forecasting fundamentals

* "Forecasting: Principles and Practice" (Hyndman & Athanasopoulos, free online, the standard textbook): <https://otexts.com/fpp3/>
* Kaggle Learn "Time Series" course: <https://www.kaggle.com/learn/time-series>
* pandas time series user guide (resampling, rolling, time zones): <https://pandas.pydata.org/docs/user_guide/timeseries.html>

## Granite TimeSeries (TTM, TSPulse, ensemble)

* granite-tsfm repository with notebooks (zero-shot, fine-tuning, exogenous): <https://github.com/ibm-granite/granite-tsfm>
* TTM-R2 model card: <https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2>
* IBM Granite time series cookbook: <https://github.com/ibm-granite-community/granite-timeseries-cookbook>
* TTM paper "Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero/Few-Shot Forecasting": <https://arxiv.org/abs/2401.03955>

## Energy forecasting

* IEA electricity reports (context on Iraq's demand growth): <https://www.iea.org/countries/iraq>
* Open Power System Data (practice datasets): <https://open-power-system-data.org/>

## IBM Power

* ppc64le/pyeco: <https://github.com/ppc64le/pyeco> · IBM wheel index: <https://wheels.developerfirst.ibm.com/ppc64le/linux>
* RocketCE conda channel: <https://anaconda.org/rocketce>
* IBM Power Virtual Server docs: <https://cloud.ibm.com/docs/power-iaas>

## Practice exercises

1. Add Iraqi public holidays and Ramadan as a 0/1 control column; measure the MAE change.
2. Forecast 10 feeders at once with `id_columns` and compare per-feeder MAE.
3. Use TSPulse to flag meter-data anomalies before training.
4. Plot forecast vs actual for the hottest week of the test period.
