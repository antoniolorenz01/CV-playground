# Computer Vision Playground

A comprehensive collection of computer vision experiments and demonstrations using state-of-the-art deep learning models. This project showcases various CV tasks including image classification, semantic segmentation, instance segmentation, attention visualization, and pose estimation.

## Overview

This repository contains Jupyter notebooks demonstrating practical implementations of modern computer vision techniques. Each section is designed to be educational and easy to follow, making it perfect for learning and experimentation.

## Features

### 1. Image Classification with Vision Transformer (ViT)
- Fine-tuning Google's ViT model on CIFAR-10 dataset
- Achieves 98% accuracy after 5 epochs
- Includes inference on custom images
- Visualizes predictions on test samples

### 2. Semantic Segmentation
- Implementation using NVIDIA's SegFormer model
- Pre-trained on ADE20K dataset (150 classes)
- Pixel-level scene understanding
- Visual comparison of original and segmented images

### 3. Attention Visualization
- Deep dive into Vision Transformer attention mechanisms
- Visualizes how the model "sees" and processes images
- Layer-by-layer attention map analysis
- Generates attention heatmaps and videos showing the [CLS] token focus

### 4. Instance Segmentation
- Mask R-CNN with ResNet-50 backbone
- Trained on COCO dataset
- Detects and segments individual object instances
- Includes bounding boxes, labels, and pixel-precise masks

### 5. Human Pose Estimation
- MediaPipe-based pose detection
- 33 body keypoint detection
- Works on both images and videos
- Real-time pose tracking capabilities

## Technologies Used

- **PyTorch** - Deep learning framework
- **Transformers** (Hugging Face) - Pre-trained transformer models
- **TorchVision** - Computer vision utilities and pre-trained models
- **MediaPipe** - Real-time pose estimation
- **Scikit-learn** - Evaluation metrics
- **Matplotlib** - Visualization
- **PIL/Pillow** - Image processing

## Installation

Install the required dependencies:

```bash
pip install torch torchvision transformers datasets matplotlib scikit-learn evaluate mediapipe opencv-python imageio imageio-ffmpeg
```

**Note:** For pose estimation, you may need specific versions:
```bash
pip install mediapipe==0.10.9 opencv-python
```

## Usage

### Running in Google Colab (Recommended)

1. Upload the notebook to Google Colab
2. Ensure GPU runtime is enabled: Runtime → Change runtime type → GPU
3. Run cells sequentially

### Running Locally

1. Clone the repository:
```bash
git clone https://github.com/yourusername/CV-playground.git
cd CV-playground
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter:
```bash
jupyter notebook notebooks/CV-playground.ipynb
```

## Project Structure

```
CV-playground/
├── notebooks/
│   └── CV-playground.ipynb    # Main notebook with all experiments
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies (if created)
```

## Results

### Image Classification
- **Accuracy:** 98% on CIFAR-10 test set
- **Model:** ViT-Base-Patch16-224
- **Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

### Semantic Segmentation
- **Model:** SegFormer-B0
- **Dataset:** ADE20K (150 semantic classes)
- **Output:** Pixel-level scene segmentation

### Instance Segmentation
- **Model:** Mask R-CNN with ResNet-50-FPN
- **Dataset:** COCO (80 object categories)
- **Features:** Object detection + instance-level segmentation masks

### Pose Estimation
- **Framework:** MediaPipe
- **Keypoints:** 33 body landmarks
- **Applications:** Image and video pose tracking

## Learning Resources

This project is perfect for:
- Understanding transformer-based vision models
- Learning different computer vision tasks
- Experimenting with pre-trained models
- Building intuition about attention mechanisms
- Exploring modern CV architectures

## Future Improvements

- [ ] Add object detection with YOLO
- [ ] Implement video classification
- [ ] Add depth estimation
- [ ] Include optical flow visualization
- [ ] Add more attention visualization techniques
- [ ] Create interactive demos with Gradio

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Add new CV experiments
- Improve documentation

## License

This project is open source and available under the MIT License.

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) for pre-trained models
- [Google Research](https://github.com/google-research/vision_transformer) for Vision Transformer
- [NVIDIA](https://github.com/NVlabs/SegFormer) for SegFormer
- [MediaPipe](https://google.github.io/mediapipe/) for pose estimation solutions
- CIFAR-10, COCO, and ADE20K datasets

---

**Note:** This project requires a GPU for optimal performance. Google Colab provides free GPU access, making it ideal for running these experiments.
