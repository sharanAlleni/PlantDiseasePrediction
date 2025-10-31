# Plant Disease Prediction System 🌿

An advanced deep learning system for detecting and classifying plant diseases using computer vision and machine learning techniques, helping farmers and agricultural professionals make informed decisions about crop health.

![TensorFlow](https://img.shields.io/badge/TensorFlow-AI%20Model-orange)
![Python](https://img.shields.io/badge/Python-Deep%20Learning-blue)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Image%20Analysis-green)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN-red)
![API](https://img.shields.io/badge/API-FastAPI-teal)

## 📊 Project Overview

The **Plant Disease Prediction System** is a sophisticated deep learning application designed to identify and diagnose plant diseases through image analysis. This project demonstrates the practical application of artificial intelligence in agriculture, helping to:

- **Early Detection** - Identify plant diseases at early stages
- **Accurate Diagnosis** - Provide precise disease classification
- **Quick Response** - Enable rapid treatment decisions
- **Cost Reduction** - Minimize crop losses and treatment expenses
- **Sustainable Farming** - Support environmentally conscious agriculture

## 🎯 Key Features

### 🤖 AI Model

- **Framework**: TensorFlow/Keras
- **Architecture**: Convolutional Neural Network (CNN)
- **Model Format**: HDF5 (.h5)
- **Deployment**: Cloud-based inference
- **Real-time Processing**: Fast image analysis

### 💻 Technical Architecture

- **Frontend**: Web-based user interface
- **Backend API**: FastAPI implementation
- **Model Serving**: TensorFlow Serving

### 📱 User Interface

- **Image Upload**: Easy-to-use image submission
- **Real-time Analysis**: Instant disease detection
- **Results Display**: Clear diagnosis presentation
- **Treatment Recommendations**: Actionable insights
- **Mobile Responsive**: Access from any device

## 🏗️ System Architecture

### Data Flow

```
Image Input → Preprocessing → Model Inference → Disease Detection → Results Display
```

### Technology Stack

- **Deep Learning**: TensorFlow, Keras
- **Backend**: FastAPI, Python
- **Frontend**: HTML, CSS, JavaScript
- **Model Serving**: TensorFlow Serving
- **Development**: Jupyter Notebooks

## 🔬 Model Architecture

### Dataset Organization

The training data is structured as follows:
```
training2/
    ├── disease_class_1/
    ├── disease_class_2/
    └── healthy/
```

### Model Configuration

The model is configured using `models.config`:
- Model Name: Plant Disease Classifier
- Platform: TensorFlow
- Model Format: SavedModel
- Version Policy: Latest Version

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- FastAPI

- Required Python packages in `requirements.txt`

### Installation

1. Clone the repository
```bash
git clone https://github.com/sharanAlleni/PlantDiseasePrediction.git
cd PlantDiseasePrediction
```

2. Install dependencies
```bash
pip install -r requirements.txt
```


3. Start the application
```bash
uvicorn api.main:app --reload
```

## 📁 Project Structure

```
PlantDiseasePrediction/
├── api/                  # FastAPI backend implementation
├── frontend/            # Web interface files
├── saved_models2/       # Trained model files
├── training2/           # Training data and notebooks
├── models.config        # Model serving configuration
└── tomatoes.h5         # Trained model for tomato diseases
```

## 🔮 Future Enhancements

### Planned Features

- **Multiple Crop Support**: Expand beyond current crop types
- **Mobile App**: Native mobile applications
- **Offline Mode**: Local model inference capability
- **Multi-language Support**: International accessibility
- **Advanced Analytics**: Disease trend analysis

### Research Directions

- **Model Optimization**: Improved accuracy and efficiency
- **Transfer Learning**: Adaptation to new crop types
- **Federated Learning**: Distributed model training
- **Explainable AI**: Better result interpretation

## 🌍 Environmental Impact

### Sustainable Agriculture

- **Reduced Chemical Usage**: Precise treatment recommendations
- **Water Conservation**: Optimal resource utilization
- **Biodiversity Protection**: Minimal environmental impact
- **Sustainable Farming**: Support for organic farming practices

## 📊 Technical Details

### Model Specifications

- **Input Format**: RGB images (224x224x3)
- **Framework**: TensorFlow 2.x
- **Architecture**: CNN with transfer learning
- **Model Size**: 2.3MB (compressed)
- **Inference Time**: ~200ms per image

### API Endpoints

- `/predict`: Image upload and analysis
- `/health`: System health check
- `/models`: Available model information
- `/stats`: Usage statistics

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

> 🌱 Growing better crops through technology!
> 
> ⭐ If you find this project helpful, please give it a star!
