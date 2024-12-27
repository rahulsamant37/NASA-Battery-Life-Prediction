# Battery Capacity Prediction System 🔋

## Overview
This project implements a machine learning pipeline for predicting battery capacity using various input parameters. The system includes data processing, model training, and a Flask web interface for making real-time predictions.

## 🌟 Key Features
- End-to-end ML pipeline for battery capacity prediction
- Real-time prediction API using Flask
- Comprehensive data validation and transformation
- MLflow integration for experiment tracking
- DagsHub integration for model registry
- Modular and maintainable code structure
- Logging and error handling

## 🏗️ Project Structure
```
Battery_Capacity_Prediction/
├── README.md
├── setup.py
├── src/
│   ├── __init__.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── data_validation.py
│   │   ├── model_evaluation.py
│   │   └── model_trainer.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── configuration.py
│   ├── constants/
│   │   └── __init__.py
│   ├── entity/
│   │   ├── __init__.py
│   │   └── config_entity.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── data_ingestion_pipeline.py
│   │   ├── data_transformation_pipeline.py
│   │   ├── data_validation_pipeline.py
│   │   ├── model_evaluation_pipeline.py
│   │   ├── model_training_pipeline.py
│   │   └── prediction_pipeline.py
│   └── utils/
│       ├── __init__.py
│       └── common.py
└── tests/
    └── test_pipeline.py
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Battery_Capacity_Prediction.git
cd Battery_Capacity_Prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package in development mode:
```bash
pip install -e .
```

This will install the package `Battery_Capacity_Prediction_System` and all its dependencies.

## ⚙️ Configuration
The project uses a modular configuration system:
- `src/config/configuration.py`: Main configuration management
- `src/entity/config_entity.py`: Configuration entities and data structures
- `src/constants/__init__.py`: Global constants and configurations

## 🔄 Training Pipeline

### Running the Training Pipeline
To execute the complete training pipeline:
```bash
python main.py
```

This will run the following stages:
1. Data Ingestion
2. Data Validation
3. Data Transformation
4. Model Training
5. Model Evaluation

### Pipeline Components
- `data_ingestion.py`: Handles data loading and initial processing
- `data_validation.py`: Validates data quality and schema
- `data_transformation.py`: Performs feature engineering and preprocessing
- `model_trainer.py`: Implements model training logic
- `model_evaluation.py`: Evaluates model performance using MLflow

## 🌐 Web Application

### Running the Flask App
To start the prediction service:
```bash
python app.py
```

### DEMO
![Flask View](https://github.com/rahulsamant37/NASA-Battery-Life-Prediction/blob/main/output/Battery%20Capacity%20Prediction%20-%20Brave%2027-12-2024%2009_34_45.png)

The application will be available at `http://localhost:8080`

### API Endpoints
- `GET /`: Home page with prediction form
- `POST /predict`: Make predictions
- `POST /train`: Retrigger model training

## 📊 Model Tracking
This project uses MLflow for experiment tracking and DagsHub for model registry. Configure your MLflow tracking URI and DagsHub credentials in the configuration files.

## 💡 Input Features
The model accepts the following features for prediction:
- Battery Type
- Start Time
- Ambient Temperature
- Battery ID
- Test ID
- UID
- Filename
- Re (Resistance)
- Rct (Charge Transfer Resistance)

## 🧪 Testing
Run the tests using:
```bash
python -m pytest tests/
```

## 🔧 Development Workflow
To modify or extend the pipeline:
1. Update configuration in `src/config/configuration.py`
2. Define new entities in `src/

### Assignment Output
#### Task-1
![Assignment View](https://github.com/rahulsamant37/NASA-Battery-Life-Prediction/blob/main/output/output1.png)

#### Task-2
![Assignment View](https://github.com/rahulsamant37/NASA-Battery-Life-Prediction/blob/main/output/output2.png)


## License
This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.

## Contact Information
For questions or collaboration opportunities:

[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:rahulsamantcoc2@gmail.com)  [![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/rahulsamant37/)  [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rahul-samant-kb37/)