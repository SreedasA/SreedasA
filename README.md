# Sreedas Anjankudy

AI/ML and Computer Vision engineer, based in Chennai, India. Focus: object detection and image classification, and the tooling around models: labelling pipelines, evaluation, serving. Results are reported with per-class metrics and documented limitations.

## Selected work

### [ev_battery_AI](https://github.com/SreedasA/ev_battery_AI): detecting EV battery pack components
- **Task:** 8-class YOLOv9m detector for battery-pack disassembly video. Dataset: 1,356 images (951 train, 270 validation, 135 test), annotated in Roboflow.
- **Method:** WIoU v3 box-regression loss, applied to Ultralytics as a runtime patch (see the [model card](https://github.com/SreedasA/ev_battery_AI/blob/main/MODEL_CARD.md)). A 13-entry [experiment log](https://github.com/SreedasA/ev_battery_AI/blob/main/EXPERIMENTS.md) compares it with SIoU and CIoU, including runs that regressed.
- **Result:** mAP50 0.752 and mAP50-95 0.554 on 270 validation images (1,723 instances). Weakest class: `low_volt_cable`, mAP50 0.263, limited by 87 training images. Trained on a 4 GB RTX 3050 laptop GPU in 6.2 hours.
- **Tooling:** CLI video annotator with per-class confidence thresholds and tracking, FastAPI web app (upload limit, job progress, health endpoint), a pseudo-labelling script that sends fasteners and cables to human review, Dockerfile for CPU inference, Kubernetes manifests as a demonstration template.

### [Dissertation-Code-on-EV-Battery-Disassembly](https://github.com/SreedasA/Dissertation-Code-on-EV-Battery-Disassembly)
- Notebook from the Aston University dissertation: YOLOv9m baseline, SIoU loss and bicubic-upsampling variants on a 1,045-image teardown dataset.

### [Emotion-Mirror-Bpf](https://github.com/SreedasA/Emotion-Mirror-Bpf): real-time webcam emotion classifier
- MediaPipe face detection, an ONNX emotion model (`enet_b0_8_best_afew`), FastAPI with WebSocket streaming at about 10 fps.
- Probabilities are averaged over 6 frames, and the label shows "uncertain" below 30% confidence. The model's 8 classes are merged into 5, with the mapping documented.

### [Ship_classification_using_Vision_Transformer](https://github.com/SreedasA/Ship_classification_using_Vision_Transformer)
- ViT ship classifier (5 classes, Kaggle optical dataset) built as a package: Hugging Face Trainer, YAML config, CI workflow, citation file linking the associated paper.

## Publication

Co-author: *Deep Learning for Image Recognition and Classification: A Comparative Study on Optical and ISAR Datasets for Maritime and Automotive Applications*, ICKECS 2025. [IEEE Xplore](https://ieeexplore.ieee.org/document/11035465)

## Background

Postgraduate Diploma in Artificial Intelligence (Aston University) · BCA (Bengaluru City University) · Microsoft Certified: Azure AI Engineer Associate (AI-102) · Research internship, DRDO CAIR (2024)

## Stack used in these repositories

PyTorch · Ultralytics YOLO · Hugging Face Transformers and Datasets · OpenCV · MediaPipe · ONNX · scikit-learn · pandas · Matplotlib · FastAPI · WebSockets · Docker · Kubernetes manifests · GitHub Actions · Roboflow · Jupyter

## Contact

Open to AI/ML and Computer Vision roles and internships in Bengaluru, Chennai, Hyderabad and Kerala.
[LinkedIn](https://www.linkedin.com/in/sreedasa) · [Email](mailto:asreedas@gmail.com)
