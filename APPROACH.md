# How we built this entry

[Leer en español](APPROACH.es.md) · [Back to the README](README.md)

The contest asks us to classify a four-channel, 256-step signal into one of eight classes. The official harness calls `predict` once per sample and scores both the predictions and how many samples fit into its 60-second time budget. That made generalization and CPU latency equally practical concerns.

## From the data to the model

We first compared the training data with the public demo. Some training-only patterns, including channel means and the `level` metadata, were strongly associated with classes but did not carry over reliably to the demo. We therefore removed those shortcut features and built an exam-like simulator: it adds noise, spikes, missing values, changes in amplitude, and other distortions across classes. We balanced the classes during training and selected the checkpoint on simulated validation data.

We tried feature-based models and several 1D CNN configurations. The selected model, `exp033`, is a small CNN trained in PyTorch. It interpolates missing values, normalizes each channel, and combines the signal with a few auxiliary measurements. A small adjustment to the class-6 output was calibrated from the *unlabeled* public demo so that the predicted class frequency better matched the expected balanced class distribution. The public demo labels were used to assess results, not to train the network or fit that adjustment.

## Preparing the submission

We exported the chosen network and its class-6 adjustment to `models/exp033.sesgo.onnx`. The submission runs on one CPU thread with ONNX Runtime, so it does not need PyTorch at inference time. We compiled preprocessing with Numba during setup, then used ONNX Runtime I/O Binding and reusable buffers to reduce work inside each `predict` call. We checked that these speed changes preserved the predictions in the comparisons we ran.

On the **400-sample public demo**, the entry gets **338 correct (84.5%)** and a score of **3.778906**. A full 60-second run in Colab, repeatedly cycling through those same 400 samples, processed 407,260 predictions and scored 120.576. The repeated run measures sustained speed; it is not an independent quality test. We also evaluated other model ideas, but kept `exp033` as the submission because they did not provide enough reliable evidence of a better quality-and-speed tradeoff.

The [README](README.md) and [verification notebook](colab.ipynb) show how to reproduce the public-demo check from a fresh environment.
