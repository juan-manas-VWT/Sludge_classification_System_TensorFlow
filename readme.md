# Sludge Classification System

A TensorFlow-based computer vision system for real-time detection of anomalies in sludge output at wastewater treatment facilities.

## Overview

This project uses deep learning to automatically classify sludge images as "normal" or "anomaly" based on visual characteristics. The system is designed to alert operators when the sludge quality deviates from acceptable parameters.

## Features

- **Automated Classification**: Distinguishes between normal and anomalous sludge conditions
- **Time-Based Dataset Creation**: Classifies training images based on timestamps
- **Fine-Tuned CNN Models**: Uses pre-trained models (EfficientNetB0/B2, MobileNetV2/V3) optimized for sludge classification
- **Comprehensive Evaluation**: Detailed performance metrics and visualization tools
- **Simple Integration**: Easy to integrate with existing camera systems

## Project Structure

```
Sludge_classification_System/
│
├── data/                   # Data directory (created during preprocessing)
│   ├── raw/                # Original labeled images (normal/anomaly)
│   ├── train/              # Training dataset
│   ├── validation/         # Validation dataset
│   └── test/               # Test dataset
│
├── models/                 # Saved models directory
│   ├── fine_tuned/         # Fine-tuned models
│   └── evaluation_results/ # Performance metrics and visualizations
│
├── notebooks/              # Jupyter notebooks for each pipeline stage
│   ├── data_exploration.py # Dataset analysis and visualization
│   ├── preprocesing.v2.py  # Data preparation and splitting
│   ├── fine-tuning.py      # Model training and optimization
│   └── evaluacion.py       # Model evaluation and testing
│
├── dataset-maker.py        # Script for creating labeled datasets from timestamped images
│
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Dataset**
   - Place raw images in the `data/raw` directory, or
   - Use `dataset-maker.py` to automatically label timestamped images:
     ```bash
     python dataset-maker.py
     ```

3. **Run the Pipeline**
   ```bash
   python notebooks/preprocesing.v2.py  # Split data into train/val/test sets
   python notebooks/fine-tuning.py      # Train the model
   python notebooks/evaluacion.py       # Evaluate performance
   ```

4. **Use the Model**
   - The best model will be saved in `models/fine_tuned/best_model.h5`
   - Example inference code is provided in the evaluation notebook

## Requirements

- Python 3.7+
- TensorFlow 2.4+
- OpenCV 4.5+
- See `requirements.txt` for full dependencies

## Customization

The system can be adapted to different types of sludge or wastewater conditions by:
- Modifying time ranges in `dataset-maker.py`
- Adjusting model architecture in `fine-tuning.py`
- Optimizing detection threshold in evaluation

## License

[MIT](https://choosealicense.com/licenses/mit/)