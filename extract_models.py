import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# This script extracts the models from your notebook and saves them for the UI
# Run this after executing all the notebook cells

def save_models_from_notebook():
    """
    Extract and save all the trained components from your notebook
    This should be run in the same session as your notebook
    """
    
    # Model data structure from your notebook
    models_data = {
        'cluster_models': {
            0.0: {
                'features': ['Player_Encoded', 'LogAst/TO', 'DraftValue', 'LogBlk', 'LogREB'],
                'scaler_mean': None,  # Will be filled from your saved_scaler
                'scaler_scale': None,
                'avg_coefs': None,    # Will be filled from your avg_coefs
                'description': 'Big Men / Centers - High blocks, rebounds'
            },
            1.0: {
                'features': ['LogAst/TO', 'LogFT%', 'Logclose_volume_x_pct', 'LogStl', 'DraftValue', 'LogDR'],
                'scaler_mean': None,
                'scaler_scale': None, 
                'avg_coefs': None,
                'description': 'Forwards - Balanced stats, good steals/defense'
            },
            2.0: {
                'features': ['LogAst', 'LogStl', 'Logclose_volume_x_pct', 'Logthree_volume_x_pct', 'DraftValue', 'LogREB'],
                'scaler_mean': None,
                'scaler_scale': None,
                'avg_coefs': None,
                'description': 'Guards - High assists, three-point shooting'
            }
        },
        'play_style_clustering': {
            'features': ['Height_in', 'Blk', '3P/100'],
            'kmeans_centers': None,    # Will be filled from your kmeans model
            'scaler_mean': None,       # From your clustering scaler
            'scaler_scale': None
        },
        'feature_transformations': {
            'height_conversion': {
                'description': 'Convert height from feet-inches to total inches',
                'example': '6-8 -> 80 inches'
            },
            'log_transforms': {
                'description': 'Log1p transforms applied to: Ast, REB, DR, OR, Ast/TO, Blk, Stl, etc.',
                'formula': 'log(1 + x)'
            }
        }
    }
    
    print("Model extraction template created.")
    print("You need to run this in your notebook environment to fill in the actual values:")
    print("""
    # In your notebook, add this code:
    
    # Save cluster models
    for cluster in [0.0, 1.0, 2.0]:
        if cluster in models_by_cluster:
            model_data = models_by_cluster[cluster]
            models_data['cluster_models'][cluster]['scaler_mean'] = model_data['scaler'].mean_.tolist()
            models_data['cluster_models'][cluster]['scaler_scale'] = model_data['scaler'].scale_.tolist() 
            models_data['cluster_models'][cluster]['avg_coefs'] = model_data['avg_coefs'].tolist()
    
    # Save clustering model
    models_data['play_style_clustering']['kmeans_centers'] = kmeans.cluster_centers_.tolist()
    models_data['play_style_clustering']['scaler_mean'] = scaler.mean_.tolist()
    models_data['play_style_clustering']['scaler_scale'] = scaler.scale_.tolist()
    
    # Save the complete models
    import pickle
    with open('ncaab_models.pkl', 'wb') as f:
        pickle.dump(models_data, f)
    
    # Also save the dataset
    final_df_transform.to_csv('player_data.csv', index=False)
    
    print("Models saved successfully!")
    """)
    
    return models_data

if __name__ == "__main__":
    save_models_from_notebook()