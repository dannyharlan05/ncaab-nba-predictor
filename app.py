from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# Global variables for your data and models
final_df_transform = None
models_by_cluster = None

def load_data():
    """Load your exported data and models"""
    global final_df_transform, models_by_cluster
    
    try:
        # Load your dataframe
        final_df_transform = pd.read_csv('final_df_transform.csv')
        print(f"Loaded {len(final_df_transform)} players")
        
        # Load your models
        with open('models_by_cluster.pkl', 'rb') as f:
            models_by_cluster = pickle.load(f)
        print(f"Loaded models for clusters: {list(models_by_cluster.keys())}")
        
        return True
    except Exception as e:
        print(f"Error loading data: {e}")
        return False

# Your exact prediction functions from the notebook
def show_clustered_player_prediction(player_name):
    row = final_df_transform[final_df_transform["Name"] == player_name]

    if row.empty:
        return {"error": f"Player '{player_name}' not found."}

    cluster = row["PlayStyleCluster"].iloc[0]

    if cluster not in models_by_cluster:
        return {"error": f"No model found for cluster {cluster}."}

    model_data = models_by_cluster[cluster]
    features = model_data["features"]
    scaler = model_data["scaler"]
    avg_coefs = model_data["avg_coefs"]

    # Get player features and make prediction
    player_features = row[features].fillna(0)
    player_scaled = scaler.transform(player_features.values.reshape(1, -1))
    logit = np.dot(player_scaled, avg_coefs)
    prob = 1 / (1 + np.exp(-logit))
    
    # Get actual outcome if available
    actual = row["Actual"].iloc[0] if "Actual" in row.columns else "Unknown"

    return {
        "player_name": player_name,
        "cluster": float(cluster),
        "probability": float(prob[0]),
        "actual": actual,
        "success": True,
        "team": row["Team"].iloc[0] if "Team" in row.columns else "",
        "year": row["Year"].iloc[0] if "Year" in row.columns else ""
    }

def explain_manual_prediction(cluster, raw_inputs):
    if cluster not in models_by_cluster:
        return {"error": f"No model found for cluster {cluster}"}
        
    model_data = models_by_cluster[cluster]
    features = model_data["features"]
    scaler = model_data["scaler"]
    avg_coefs = model_data["avg_coefs"]

    # Build input vector
    input_vector = {feat: raw_inputs.get(feat, 0.0) for feat in features}
    df_input = pd.DataFrame([input_vector])

    # Scale input and compute prediction
    X_scaled = scaler.transform(df_input)[0]
    contributions = X_scaled * avg_coefs
    logit = np.sum(contributions)
    prob = 1 / (1 + np.exp(-logit))

    # Feature breakdown
    feature_breakdown = []
    for feat, raw_val, scaled_val, coef, contrib in zip(features, df_input.iloc[0], X_scaled, avg_coefs, contributions):
        feature_breakdown.append({
            "feature": feat,
            "raw_value": float(raw_val),
            "scaled_value": float(scaled_val),
            "coefficient": float(coef),
            "contribution": float(contrib)
        })

    result = {
        "cluster": cluster,
        "probability": float(prob),
        "logit_total": float(logit),
        "feature_breakdown": feature_breakdown,
        "success": True
    }

    # Apply class year adjustment for cluster 1.0
    if cluster == 1.0 and "Player_Encoded" in raw_inputs:
        player_encoded = raw_inputs["Player_Encoded"]
        adjustment_map = {1: 0.07, 2: -0.04, 3: -0.09, 4: -0.13}
        adjustment = adjustment_map.get(player_encoded, 0.0)
        
        if prob > 0.9 and adjustment > 0:
            adjusted_prob = prob
        else:
            adjusted_prob = max(0.0, min(1.0, prob + adjustment))
        
        result["adjusted_probability"] = float(adjusted_prob)
        result["adjustment"] = float(adjustment)

    return result

@app.route('/')
def index():
    if not load_data():
        return "Error: Could not load data. Make sure to export your models first."
    return """
    <!DOCTYPE html>
    <html>
    <head><title>NCAAB Predictor</title></head>
    <body>
        <h1>🏀 NCAAB NBA Success Predictor</h1>
        <p>Search for a player:</p>
        <input type="text" id="playerName" placeholder="Enter player name">
        <button onclick="searchPlayer()">Search</button>
        <div id="result"></div>
        
        <script>
        function searchPlayer() {
            const name = document.getElementById('playerName').value;
            fetch('/predict_player', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({player_name: name})
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('result').innerHTML = 
                    '<h3>Result:</h3><pre>' + JSON.stringify(data, null, 2) + '</pre>';
            });
        }
        </script>
    </body>
    </html>
    """

@app.route('/predict_player', methods=['POST'])
def predict_player():
    data = request.json
    player_name = data.get('player_name', '').strip()
    
    if not player_name:
        return jsonify({"success": False, "error": "Player name required"})
    
    result = show_clustered_player_prediction(player_name)
    return jsonify(result)

@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    data = request.json
    cluster = float(data.get('cluster', 1.0))
    
    # Extract input values for the features
    raw_inputs = {}
    for key, value in data.items():
        if key != 'cluster':
            try:
                raw_inputs[key] = float(value)
            except:
                raw_inputs[key] = 0.0
    
    result = explain_manual_prediction(cluster, raw_inputs)
    return jsonify(result)

@app.route('/search_suggestions')
def search_suggestions():
    query = request.args.get('q', '')
    if len(query) < 2 or final_df_transform is None:
        return jsonify([])
    
    matches = final_df_transform[
        final_df_transform["Name"].str.contains(query, case=False, na=False)
    ]["Name"].head(10).tolist()
    return jsonify(matches)

@app.route('/get_cluster_info/<float:cluster>')
def get_cluster_info(cluster):
    if models_by_cluster is None or cluster not in models_by_cluster:
        return jsonify({"error": "Cluster not found"})
    
    features = models_by_cluster[cluster]["features"]
    descriptions = {
        0.0: "Big Men/Centers - High blocks and rebounds",
        1.0: "Forwards - Balanced stats, good defense", 
        2.0: "Guards - High assists and three-point shooting"
    }
    
    return jsonify({
        "cluster": cluster,
        "description": descriptions.get(cluster, "Unknown"),
        "features": features
    })

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')