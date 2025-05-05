# Anomaly Detection System for Veolia Kruger

This project implements a real-time anomaly detection system to monitor sludge output at the Veolia Kruger industrial plant using Computer Vision techniques and TensorFlow.

## Description

The system detects anomalies in sludge output, such as cracks or conglomerated parts that are undesirable, using a fine-tuned deep learning model. When an anomaly is detected, the system sends alerts to operators so they can take corrective actions.

## Project Structure

```
veolia_anomaly_detection/
│
├── data/
│   ├── raw/                # Original images
│   ├── processed/          # Preprocessed images
│   ├── train/              # Training dataset
│   └── validation/         # Validation dataset
│
├── models/
│   ├── pretrained/         # Pre-trained models
│   ├── fine_tuned/         # Models after fine-tuning
│   └── evaluation_results/ # Evaluation results
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_fine_tuning.ipynb
│   └── 04_evaluation.ipynb
│
├── src/
│   ├── data/               # Data handling functions
│   ├── models/             # Model definition and training
│   └── utils/              # General utilities
│
├── inference/
│   ├── realtime_detection.py  # Script for real-time detection
│   └── alert_system.py        # Alert system
│
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```

## Requirements

- Python 3.7+
- TensorFlow 2.4+ 
- OpenCV 4.5+
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/veolia-kruger/anomaly-detection.git
   cd anomaly-detection
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Workflow

### 1. Data Exploration

The notebook `01_data_exploration.ipynb` allows you to explore and visualize the images to better understand the characteristics of normal and anomalous samples.

### 2. Preprocessing

The notebook `02_preprocessing.ipynb` performs image preprocessing, including resizing, normalization, and splitting into training and validation sets.

### 3. Training and Fine-Tuning

The notebook `03_fine_tuning.ipynb` implements fine-tuning of a pre-trained model (such as EfficientNetB0, MobileNetV2, or ResNet50) to adapt it to the detection of specific anomalies.

### 4. Evaluation

The notebook `04_evaluation.ipynb` evaluates the model's performance using metrics such as precision, recall, ROC curves, and optimal threshold analysis.

### 5. Real-Time Detection

The `realtime_detection.py` script implements real-time detection using the trained model. It can process video from a live camera or a video file.

```bash
# Example usage with webcam
python inference/realtime_detection.py --source 0

# Example with RTSP stream
python inference/realtime_detection.py --source rtsp://example.com/stream

# Example with video file
python inference/realtime_detection.py --source video.mp4

# Customize detection threshold
python inference/realtime_detection.py --source 0 --threshold 0.75

# Activate alert system
python inference/realtime_detection.py --source 0 --alerts
```

## Alert System

The `alert_system.py` module provides functionalities to notify when an anomaly is detected. It supports multiple notification methods:

- Local event logging
- Email notifications
- Text messages (SMS)
- Push notifications
- Integration with Veolia Kruger management systems

To configure the alert system:

```bash
# Create an example configuration file
python inference/alert_system.py --create-config

# Edit the resulting file (alert_config.json) with your credentials
```

## Model Customization

To customize the model for a specific application:

1. Collect representative images of normal and anomalous conditions.
2. Organize them in the `data/raw/normal` and `data/raw/anomaly` folders.
3. Run notebooks 1 through 4 to train and evaluate your custom model.
4. Use the resulting model with the real-time detection script.

## Contributions

Contributions are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
