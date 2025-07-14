# Run this code in your Jupyter notebook to export the models and data for the UI

# Add this cell to your notebook and run it:

"""
import pickle

# Export your final dataframe
final_df_transform.to_csv('final_df_transform.csv', index=False)
print("Exported final_df_transform.csv")

# Export your models dictionary
with open('models_by_cluster.pkl', 'wb') as f:
    pickle.dump(models_by_cluster, f)
print("Exported models_by_cluster.pkl")

print("Ready to run the Flask app!")
"""

print("Copy and paste the code above into a new cell in your notebook and run it.")
print("This will create the files needed for the web UI.")