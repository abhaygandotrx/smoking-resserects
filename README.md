









---

# Smoking Resserects

Smoking Resserects is a Python-based smoke detection and risk assessment project.
The repository contains scripts for training a model, detecting smoke using images or camera input, and estimating potential risk based on detected patterns.

This project is suitable for academic demos, research prototypes, and early-stage safety or health monitoring systems.

---

## Project Overview

The goal of this project is to:

* Detect the presence of smoke using machine learning techniques
* Run real-time or file-based detection
* Estimate risk levels using a separate risk model
* Provide a modular structure that can be extended further

---

## Repository Structure

| File                | Description                                          |
| ------------------- | ---------------------------------------------------- |
| `app.py`            | Main application entry point                         |
| `camera_demo.py`    | Runs smoke detection using live camera or video feed |
| `create_model.py`   | Creates and initializes the smoke detection model    |
| `train_model_4f.py` | Trains the model using the provided dataset          |
| `detector.py`       | Core smoke detection logic                           |
| `risk_model.py`     | Calculates risk level based on detection output      |
| `demo_data.csv`     | Sample dataset for training and testing              |
| `requirements.txt`  | List of required Python dependencies                 |

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/abhaygandotrx/smoking-resserects.git
cd smoking-resserects
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Run camera-based smoke detection

```bash
python camera_demo.py
```

### Train the model

```bash
python train_model_4f.py
```

### Create or configure a model

```bash
python create_model.py
```

### Run the main application

```bash
python app.py
```

---

## Dataset

* `demo_data.csv` is provided as a sample dataset.
* You can replace it with your own dataset for better accuracy.
* Ensure proper preprocessing before training.

---

## Future Improvements

* Improve model accuracy with larger datasets
* Add evaluation metrics and visualization
* Build a web or desktop interface
* Integrate alert systems (email, SMS, notifications)
* Optimize performance for real-time deployment

---

## License

No license has been added yet.
Add a license file if you plan to make this project open-source.

---

If you want, next we can:

* Tighten this for internship-ready GitHub vibes
* Add a proper ML workflow section
* Or rewrite it to sound research-paper level instead of student level
