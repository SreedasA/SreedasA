# Sreedas Anjankudy

**Computer vision engineer. I take detection models from labelled data to a served API, and I document where they fail.**

Chennai, India · Open to AI/ML and Computer Vision roles and internships in Bengaluru, Chennai, Hyderabad and Kerala
[LinkedIn](https://www.linkedin.com/in/sreedasa) · [Email](mailto:asreedas@gmail.com)

**Start here:** [ev_battery_AI](https://github.com/SreedasA/ev_battery_AI), then its [model card](https://github.com/SreedasA/ev_battery_AI/blob/main/MODEL_CARD.md) and [experiment log](https://github.com/SreedasA/ev_battery_AI/blob/main/EXPERIMENTS.md).

<a href="https://github.com/SreedasA/ev_battery_AI"><img src="https://raw.githubusercontent.com/SreedasA/ev_battery_AI/main/docs/demo.gif" alt="YOLOv9m detecting EV battery components in disassembly footage" width="640"></a>

## Flagship: EV battery component detector

- **Problem:** locate 8 kinds of components (modules, busbars, coolant tubes, fasteners, cables and others) in battery-pack disassembly video.
- **Approach:** YOLOv9m with a WIoU v3 box-regression loss, applied to Ultralytics as a runtime patch. Dataset: 1,356 images (951 train, 270 validation, 135 test), annotated in Roboflow. A 13-entry experiment log compares WIoU v3 with SIoU and CIoU, with every config and decision recorded.
- **Result:** mAP50 0.752 and mAP50-95 0.554 on 270 validation images (1,723 instances), trained in 6.2 hours on a 4 GB RTX 3050 laptop GPU.
- **Where it fails:** `low_volt_cable` scores mAP50 0.263 (87 training images; the validation set has only 20 images with cables). Training longer made it worse (0.303 at 200 epochs, 0.263 at 250). `fasteners` recall is 0.590.
- **Engineering:** CLI video annotator with per-class confidence thresholds and tracking; FastAPI web app with upload limit, job progress and health endpoint; pseudo-labelling script that sends fasteners and cables to human review; Dockerfile for CPU inference; Kubernetes manifests as a demonstration template.
- **Current focus:** the weakest class, through a larger reviewed dataset and sliced inference, as planned in the repository's limitations section.

## Other projects

- [**Emotion-Mirror-Bpf**](https://github.com/SreedasA/Emotion-Mirror-Bpf): real-time webcam emotion classifier. MediaPipe face detection, an ONNX emotion model, FastAPI with WebSocket streaming at about 10 fps. Predictions are averaged over 6 frames and shown as "uncertain" below 30% confidence.
- [**Ship_classification_using_Vision_Transformer**](https://github.com/SreedasA/Ship_classification_using_Vision_Transformer): ViT ship classifier (5 classes, Kaggle optical dataset) packaged with Hugging Face Trainer, YAML config and a CI workflow.
- [**Dissertation-Code-on-EV-Battery-Disassembly**](https://github.com/SreedasA/Dissertation-Code-on-EV-Battery-Disassembly): notebook from the Aston University dissertation comparing a YOLOv9m baseline, SIoU loss and bicubic-upsampling variants on a 1,045-image teardown dataset.

## Background

- Postgraduate Diploma in Artificial Intelligence, Aston University · BCA, Bengaluru City University
- Microsoft Certified: Azure AI Engineer Associate (AI-102)
- Research internship, DRDO CAIR (2024)
- Co-author, ICKECS 2025: [*Deep Learning for Image Recognition and Classification: A Comparative Study on Optical and ISAR Datasets for Maritime and Automotive Applications*](https://ieeexplore.ieee.org/document/11035465)

## Tools used in these repositories

Python · PyTorch · Ultralytics YOLO · Hugging Face Transformers · OpenCV · MediaPipe · FastAPI · Docker · Roboflow
