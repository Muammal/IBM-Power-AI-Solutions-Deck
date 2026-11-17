# Solution 10 — Learning resources

## Computer vision fundamentals

* Hugging Face Computer Vision Course (free): <https://huggingface.co/learn/computer-vision-course>
* Stanford CS231n notes (CNNs, detection): <https://cs231n.github.io/>
* OpenCV Python tutorials: <https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html>

## Object detection with transformers

* RT-DETR paper "DETRs Beat YOLOs on Real-time Object Detection": <https://arxiv.org/abs/2304.08069>
* RT-DETRv2 model card and checkpoints: <https://huggingface.co/PekingU/rtdetr_v2_r18vd>
* Transformers object-detection fine-tuning guide: <https://huggingface.co/docs/transformers/tasks/object_detection>
* Evaluation metrics explained (precision, recall, AP): <https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173>

## Data labelling

* CVAT (open source, COCO export): <https://docs.cvat.ai/>
* Label Studio: <https://labelstud.io/guide/>

## Video engineering

* PyAV documentation: <https://pyav.basswood-io.com/docs/stable/>
* RTSP / ONVIF basics for IP cameras: <https://www.onvif.org/profiles/>

## Deploying models with ONNX Runtime

* PyTorch → ONNX (dynamo exporter): <https://docs.pytorch.org/docs/stable/onnx.html>
* ONNX Runtime performance tuning (threads): <https://onnxruntime.ai/docs/performance/tune-performance/>
* Prometheus Python client: <https://prometheus.github.io/client_python/>

## Responsible use

* UK Surveillance Camera Code of Practice (a widely used reference): <https://www.gov.uk/government/publications/update-to-surveillance-camera-code>
* NIST AI Risk Management Framework: <https://www.nist.gov/itl/ai-risk-management-framework>

## IBM Power

* ppc64le/pyeco: <https://github.com/ppc64le/pyeco> · IBM wheel index: <https://wheels.developerfirst.ibm.com/ppc64le/linux>
* RocketCE conda channel: <https://anaconda.org/rocketce>
* IBM Power Virtual Server docs: <https://cloud.ibm.com/docs/power-iaas>

## Practice exercises

1. Add a per-camera schedule (analyse a loading dock only outside working hours).
2. Add a "loitering near entrance" rule using the tracker's track duration for `person`.
3. Export the detector at 512×512 and compare speed vs recall on your test set.
4. Build a Grafana dashboard from `/metrics`: alerts per camera per day, camera uptime, inference latency.
