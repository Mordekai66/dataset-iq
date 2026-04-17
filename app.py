import os
import json
import pandas as pd
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)
DATA_DIR = "data/ml"

def get_available_datasets():
    datasets = []
    if not os.path.exists(DATA_DIR):
        return datasets
    
    files = os.listdir(DATA_DIR)
    
    data_files = [f for f in files if f.endswith(('.csv', '.xlsx', '.xls'))]
    
    for data_file in data_files:
        base_name = os.path.splitext(data_file)[0]
        stats_file = f"{base_name}.stats.json"
        
        dataset_info = {
            'name': base_name,
            'data_file': data_file,
            'stats_file': stats_file,
            'has_stats': stats_file in files
        }
        
        if dataset_info['has_stats']:
            try:
                with open(os.path.join(DATA_DIR, stats_file), 'r', encoding='utf-8') as f:
                    stats_data = json.load(f)
                    dataset_info['summary'] = stats_data.get('summary', {})
                    dataset_info['issues'] = stats_data.get('issues', {})
            except Exception as e:
                dataset_info['error'] = str(e)
        
        datasets.append(dataset_info)
    
    return datasets

def load_dataset_data(dataset_name):
    for ext in ['.csv', '.xlsx', '.xls']:
        file_path = os.path.join(DATA_DIR, f"{dataset_name}{ext}")
        if os.path.exists(file_path):
            try:
                if ext == '.csv':
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                
                preview = df.head(100).to_dict(orient='records')
                columns = df.columns.tolist()
                
                return {
                    'columns': columns,
                    'preview': preview,
                    'total_rows': len(df),
                    'total_columns': len(columns)
                }
            except Exception as e:
                return {'error': str(e)}
    
    return {'error': 'Dataset not found'}

@app.route('/')
def index():
    datasets = get_available_datasets()
    return render_template('index.html', datasets=datasets)

@app.route('/api/datasets')
def api_datasets():
    datasets = get_available_datasets()
    return jsonify(datasets)

@app.route('/api/dataset/<dataset_name>')
def api_dataset_detail(dataset_name):
    data = load_dataset_data(dataset_name)
    return jsonify(data)

@app.route('/api/stats/<dataset_name>')
def api_dataset_stats(dataset_name):
    stats_file = os.path.join(DATA_DIR, f"{dataset_name}.stats.json")
    if os.path.exists(stats_file):
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        return jsonify(stats)
    return jsonify({'error': 'Stats not found'}), 404

@app.route('/dataset/<dataset_name>')
def dataset_view(dataset_name):
    datasets = get_available_datasets()
    current_dataset = next((d for d in datasets if d['name'] == dataset_name), None)
    
    if not current_dataset:
        return render_template('404.html'), 404
    
    return render_template('dataset.html', dataset=current_dataset, dataset_name=dataset_name)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)