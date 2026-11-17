# Solution 12 — Learning resources

## ANPR and OCR

* CRNN paper "An End-to-End Trainable Neural Network for Image-based Sequence Recognition" (Shi et al.): <https://arxiv.org/abs/1507.05717>
* CTC explained (Distill, "Sequence Modeling with CTC"): <https://distill.pub/2017/ctc/>
* PyTorch CTCLoss documentation: <https://docs.pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html>
* Hugging Face Computer Vision Course (detection, data augmentation): <https://huggingface.co/learn/computer-vision-course>

## Detection and tracking

* RT-DETRv2 checkpoints: <https://huggingface.co/PekingU/rtdetr_v2_r18vd>
* Multi-object tracking basics (SORT paper): <https://arxiv.org/abs/1602.00763>
* OpenCV geometric transformations / homography: <https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html>

## Speed measurement and time

* Average speed enforcement overview (European Road Safety Observatory): <https://road-safety.transport.ec.europa.eu/>
* chrony documentation (NTP, accuracy, monitoring): <https://chrony-project.org/documentation.html>
* OIML R 91 (radar speed meters) as an example of metrological requirements for speed equipment: <https://www.oiml.org/en/publications/recommendations>

## Evidence and cryptography

* Ed25519 signatures with Python `cryptography`: <https://cryptography.io/en/latest/hazmat/primitives/asymmetric/ed25519/>
* WHO Global status report on road safety (context for Iraq): <https://www.who.int/teams/social-determinants-of-health/safety-and-mobility/global-status-report-on-road-safety-2023>

## IBM Power

* ppc64le/pyeco: <https://github.com/ppc64le/pyeco> · IBM wheel index: <https://wheels.developerfirst.ibm.com/ppc64le/linux>
* RocketCE conda channel: <https://anaconda.org/rocketce>
* IBM Power Virtual Server docs: <https://cloud.ibm.com/docs/power-iaas>

## Practice exercises

1. Add a red-light violation type: stop-line crossing while a signal-state input is red.
2. Show travel-time statistics (`/api/journeys`) on a map of the sections, refreshing every 15 minutes.
3. Measure how OCR accuracy changes with plate width in pixels on your test set; set a minimum crop width.
4. Write a verifier script for evidence packages that a court expert could run (manifest + signature + hashes).
