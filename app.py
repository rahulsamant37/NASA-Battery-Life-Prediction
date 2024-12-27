from flask import Flask, render_template, request, jsonify
import pandas as pd
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.pipeline.model_training_pipeline import ModelTrainingPipeline
from src.pipeline.model_evaluation_pipeline import ModelEvaluationPipeline

app = Flask(__name__)

# Route to render the prediction page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'type': [request.form['type']],
            'start_time': [request.form['start_time']],
            'ambient_temperature': [float(request.form['ambient_temperature'])],
            'battery_id': [request.form['battery_id']],
            'test_id': [int(request.form['test_id'])],
            'uid': [int(request.form['uid'])],
            'filename': [request.form['filename']],
            'Re': [float(request.form['Re'])],
            'Rct': [float(request.form['Rct'])],
        }

        input_data = pd.DataFrame(data)
        pipeline = PredictionPipeline()
        capacity = pipeline.predict(input_data)
        
        return jsonify({
            'status': 'success',
            'predicted_capacity': f"{capacity:.4f}"
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

# Route to handle retraining requests
@app.route('/train', methods=['POST'])
def train():
    try:
        pipeline = ModelTrainingPipeline()
        pipeline.inititate_model_training()
        eval_pipeline = ModelEvaluationPipeline()
        eval_pipeline.inititate_model_evaluation()
        
        return jsonify({
            'status': 'success',
            'message': 'Model retrained successfully!'
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080)