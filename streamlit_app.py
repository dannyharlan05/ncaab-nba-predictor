import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page config
st.set_page_config(page_title="NCAAB NBA Success Predictor", page_icon="🏀")

# Title
st.title("🏀 NCAAB NBA Success Predictor")
st.write("Predict NBA success probability (VORP > 4 in first 4 seasons) using college basketball stats")

# Load data (with caching for performance)
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('final_df_transform.csv')
        with open('models_by_cluster.pkl', 'rb') as f:
            models = pickle.load(f)
        return df, models
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

final_df_transform, models_by_cluster = load_data()

if final_df_transform is None:
    st.error("Could not load data. Make sure to export your models from the notebook first.")
    st.stop()

# Success message
st.success(f"✅ Loaded {len(final_df_transform)} players and {len(models_by_cluster)} models!")

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["About", "Player Search", "Player Comparison", "Rankings & Analysis", "Draft Class Analysis", "Player Rankings"])

with tab1:
    # Header with visual styling
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(90deg, #1f77b4, #ff7f0e); border-radius: 10px; margin-bottom: 30px;'>
        <h1 style='color: white; margin: 0; font-size: 2.5rem;'>NCAAB NBA Success Predictor</h1>
        <p style='color: white; margin: 10px 0 0 0; font-size: 1.2rem;'>Machine Learning Model for Basketball Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats overview cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; border-left: 4px solid #2ca02c;'>
            <h3 style='color: #2ca02c; margin: 0;'>2010-2018</h3>
            <p style='margin: 5px 0 0 0;'>Training Period</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; border-left: 4px solid #d62728;'>
            <h3 style='color: #d62728; margin: 0;'>2019-2025</h3>
            <p style='margin: 5px 0 0 0;'>Testing Period</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main content with better styling
    st.markdown("""
    <div style='background-color: white; padding: 30px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin: 20px 0;'>
        <h2 style='color: #1f77b4; border-bottom: 2px solid #e1e5e9; padding-bottom: 10px;'>📊 Overview</h2>
        <p style='font-size: 1.1rem; line-height: 1.6; color: #333;'>
            This machine learning model predicts <strong>NBA success in the first four years</strong> of a player's career based on their college basketball statistics and performance. Important to note that if the player either did not play college basketball, or did not play 15 games of college basketball, they will not have a prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Usage tip
    st.markdown("""
    <div style='background-color: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107; margin: 20px 0;'>
        <strong style='color: #856404;'>💡 Usage Tip:</strong> When you first interact with any feature (search, filters, etc.), the page will reset to this home tab. After that first interaction, you can navigate freely between tabs without any resets.
    </div>
    """, unsafe_allow_html=True)
    

with tab2:
    # Header with styling
    st.markdown("""
    <div style='text-align: center; padding: 15px; background: linear-gradient(90deg, #1f77b4, #2ca02c); border-radius: 10px; margin-bottom: 20px;'>
        <h1 style='color: white; margin: 0;'>Player Search & Analysis</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Add model training info with better styling
    st.markdown("""
    <div style='background-color: #e7f3ff; padding: 15px; border-radius: 8px; border-left: 4px solid #1f77b4; margin-bottom: 20px;'>
        <strong style='color: #1f77b4;'>Model Training:</strong> This model was trained on data from 2010-2018<br>
        <strong style='color: #1f77b4;'>Ratings:</strong> Rating calculated and available for players drafted from 2019-2025
    </div>
    """, unsafe_allow_html=True)
    
    # Player search with autocomplete
    player_name = st.text_input("Enter player name:", placeholder="e.g., Shai Gilgeous-Alexander")
    
    if player_name:
        # Find matching players (only 2019-2025)
        matches = final_df_transform[
            (final_df_transform["Name"].str.contains(player_name, case=False, na=False)) &
            (final_df_transform['Year'] >= 19) & 
            (final_df_transform['Year'] <= 25)
        ]
        
        if len(matches) == 0:
            st.warning(f"No players found matching '{player_name}' in the 2019-2025 dataset")
        elif len(matches) == 1:
            # Exact match - make prediction
            player = matches.iloc[0]
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Player Info:**")
                st.write(f"**Name:** {player.get('Name', 'Unknown')}")
                st.write(f"**Team:** {player.get('Team', 'Unknown')}")
                st.write(f"**Draft Pick:** #{int(player.get('Pick', 0)) if player.get('Pick', 0) > 0 else 'Undrafted'}")
                st.write(f"**Height:** {player.get('Height', 'Unknown')}")
            
            with col2:
                st.write("**Key Stats:**")
                st.write(f"**BPM:** {player.get('BPM', 0):.1f}")
                st.write(f"**OREB%+DREB%:** {player.get('REB', 0):.1f}")
                st.write(f"**Ast%:** {player.get('Ast', 0):.1f}")
                st.write(f"**Blk%:** {player.get('Blk', 0):.1f}")
            
            # Make prediction
            cluster = player["PlayStyleCluster"]
            if cluster in models_by_cluster:
                model_data = models_by_cluster[cluster]
                features = model_data["features"]
                scaler = model_data["scaler"]
                avg_coefs = model_data["avg_coefs"]
                
                # Get player features and make prediction
                player_features = player[features].fillna(0)
                player_scaled = scaler.transform(player_features.values.reshape(1, -1))
                logit = np.dot(player_scaled, avg_coefs)
                prob = 1 / (1 + np.exp(-logit))
                
                
                
                # Display prediction
                st.markdown("---")
                st.subheader("Prediction Results")
                
                # Additional info
                
                # Calculate rating within position using 2010-2025 data for ratings
                player_year = player.get('Year', 0)
                
                # Get ALL 2010-2025 players in same position for rating calculation
                cluster_players = final_df_transform[
                    (final_df_transform['PlayStyleCluster'] == cluster) & 
                    (final_df_transform['Year'] >= 10) & 
                    (final_df_transform['Year'] <= 25)
                ]
                
                if len(cluster_players) > 0:
                    # Calculate predictions for all players in this position
                    predictions = []
                    for idx, row in cluster_players.iterrows():
                        try:
                            row_features = row[features].fillna(0)
                            row_scaled = scaler.transform(row_features.values.reshape(1, -1))
                            row_logit = np.dot(row_scaled, avg_coefs)
                            row_prob = 1 / (1 + np.exp(-row_logit))
                            
                            predictions.append(row_prob[0])
                        except:
                            predictions.append(0)
                    
                    cluster_players = cluster_players.copy()
                    cluster_players['Prediction'] = predictions
                    cluster_players = cluster_players.sort_values('Prediction', ascending=False).reset_index(drop=True)
                    
                    # Calculate player's rating within their position
                    player_rank_idx = cluster_players[cluster_players['Name'] == player['Name']].index
                    if len(player_rank_idx) > 0:
                        rank = player_rank_idx[0] + 1  # 1-based rank
                        total = len(cluster_players)
                        # Calculate rating (prediction score)
                        rating = prob[0]
                    else:
                        rating = 0
                else:
                    rating = 0
                
                # Color based on rating - 8 granular categories
                if rating >= 0.9:
                    color = "#006400"
                    bg_color = "#d4edda"
                    badge = "ELITE"
                elif rating >= 0.8:
                    color = "#228b22"
                    bg_color = "#d4edda"
                    badge = "GREAT"
                elif rating >= 0.7:
                    color = "#32cd32"
                    bg_color = "#e8f5e8"
                    badge = "VERY GOOD"
                elif rating >= 0.6:
                    color = "#9acd32"
                    bg_color = "#f0f8e8"
                    badge = "GOOD"
                elif rating >= 0.5:
                    color = "#ff7f0e"
                    bg_color = "#fff3cd"
                    badge = "ABOVE AVERAGE"
                elif rating >= 0.4:
                    color = "#ffa500"
                    bg_color = "#fff8dc"
                    badge = "AVERAGE"
                elif rating >= 0.3:
                    color = "#ff6347"
                    bg_color = "#ffe4e1"
                    badge = "BELOW AVERAGE"
                else:
                    color = "#d62728"
                    bg_color = "#f8d7da"
                    badge = "POOR"
                
                st.markdown(f"""
                <div style='background-color: {bg_color}; padding: 20px; border-radius: 10px; text-align: center; margin: 15px 0; border: 2px solid {color};'>
                    <h2 style='color: {color}; margin: 0; font-size: 2.5rem;'>{rating:.3f}</h2>
                    <p style='color: {color}; margin: 5px 0; font-size: 1.1rem; font-weight: bold;'>RATING</p>
                    <span style='background-color: {color}; color: white; padding: 5px 15px; border-radius: 20px; font-size: 0.9rem; font-weight: bold;'>{badge}</span>
                </div>
                """, unsafe_allow_html=True)
                
        
        else:
            # Multiple matches - show options
            st.write(f"Found {len(matches)} players matching '{player_name}'. Please be more specific or use the exact name.")

with tab3:
    # Header with styling
    st.markdown("""
    <div style='text-align: center; padding: 15px; background: linear-gradient(90deg, #ff7f0e, #d62728); border-radius: 10px; margin-bottom: 20px;'>
        <h1 style='color: white; margin: 0;'>Player Comparison</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Add model training info
    st.markdown("""
    <div style='background-color: #e7f3ff; padding: 15px; border-radius: 8px; border-left: 4px solid #1f77b4; margin-bottom: 20px;'>
        <strong style='color: #1f77b4;'>Note:</strong> Only players from 2019-2025 are available for search and comparison. Players are ranked by their prediction scores.
    </div>
    """, unsafe_allow_html=True)
    
    # Two column layout for player inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Player 1")
        player1_name = st.text_input("Enter first player name:", placeholder="e.g., Shai Gilgeous-Alexander", key="comp_player1")
    
    with col2:
        st.markdown("### Player 2") 
        player2_name = st.text_input("Enter second player name:", placeholder="e.g., Luka Doncic", key="comp_player2")
    
    # Only show comparison when both players are entered
    if player1_name and player2_name:
        # Find both players (only 2019-2025)
        matches1 = final_df_transform[
            (final_df_transform["Name"].str.contains(player1_name, case=False, na=False)) &
            (final_df_transform['Year'] >= 19) & 
            (final_df_transform['Year'] <= 25)
        ]
        matches2 = final_df_transform[
            (final_df_transform["Name"].str.contains(player2_name, case=False, na=False)) &
            (final_df_transform['Year'] >= 19) & 
            (final_df_transform['Year'] <= 25)
        ]
        
        if len(matches1) == 0:
            st.warning(f"No players found matching '{player1_name}' in the 2019-2025 dataset")
        elif len(matches2) == 0:
            st.warning(f"No players found matching '{player2_name}' in the 2019-2025 dataset")
        elif len(matches1) > 1:
            st.warning(f"Multiple players found matching '{player1_name}'. Please be more specific.")
        elif len(matches2) > 1:
            st.warning(f"Multiple players found matching '{player2_name}'. Please be more specific.")
        else:
            # Get the single match for each player
            player1 = matches1.iloc[0]
            player2 = matches2.iloc[0]
            
            st.markdown("---")
            st.subheader("Comparison Results")
            
            # Display both players side by side using same logic as player search
            col1, col2 = st.columns(2)
            
            # Player 1 column
            with col1:
                st.markdown(f"### {player1['Name']}")
                
                # Basic info
                st.write(f"**Team:** {player1.get('Team', 'Unknown')}")
                st.write(f"**Draft Pick:** #{int(player1.get('Pick', 0)) if player1.get('Pick', 0) > 0 else 'Undrafted'}")
                st.write(f"**Height:** {player1.get('Height', 'Unknown')}")
                
                # Key stats
                st.write("**Key Stats:**")
                st.write(f"**BPM:** {player1.get('BPM', 0):.1f}")
                st.write(f"**OREB%+DREB%:** {player1.get('REB', 0):.1f}")
                st.write(f"**Ast%:** {player1.get('Ast', 0):.1f}")
                st.write(f"**Blk%:** {player1.get('Blk', 0):.1f}")
                
                # Calculate prediction score
                cluster1 = player1["PlayStyleCluster"]
                if cluster1 in models_by_cluster:
                    # Get 2019-2025 players in same cluster for comparison
                    cluster_players = final_df_transform[
                        (final_df_transform['PlayStyleCluster'] == cluster1) & 
                        (final_df_transform['Year'] >= 10) & 
                        (final_df_transform['Year'] <= 25)
                    ]
                    
                    if len(cluster_players) > 0:
                        # Calculate predictions for all players in cluster
                        model_data = models_by_cluster[cluster1]
                        features = model_data["features"]
                        scaler = model_data["scaler"]
                        avg_coefs = model_data["avg_coefs"]
                        
                        # Get player1's prediction
                        player1_features = player1[features].fillna(0)
                        player1_scaled = scaler.transform(player1_features.values.reshape(1, -1))
                        logit1 = np.dot(player1_scaled, avg_coefs)
                        prob1 = 1 / (1 + np.exp(-logit1))
                        
                        
                        # Calculate all cluster predictions for ranking
                        predictions = []
                        for idx, row in cluster_players.iterrows():
                            try:
                                row_features = row[features].fillna(0)
                                row_scaled = scaler.transform(row_features.values.reshape(1, -1))
                                row_logit = np.dot(row_scaled, avg_coefs)
                                row_prob = 1 / (1 + np.exp(-row_logit))
                                
                                predictions.append(row_prob[0])
                            except:
                                predictions.append(0)
                        
                        cluster_players = cluster_players.copy()
                        cluster_players['Prediction'] = predictions
                        cluster_players = cluster_players.sort_values('Prediction', ascending=False).reset_index(drop=True)
                        
                        # Check if player1 is in 2019-2025
                        player1_year = player1.get('Year', 0)
                        if 19 <= player1_year <= 24:
                            # Player is in 2019-2025, show actual rank
                            player1_rank_idx = cluster_players[cluster_players['Name'] == player1['Name']].index
                            if len(player1_rank_idx) > 0:
                                rank1 = player1_rank_idx[0] + 1
                                total1 = len(cluster_players)
                        # Calculate player1's rating (prediction score)
                        player1_year = player1.get('Year', 0)
                        rating1 = prob1[0]
                        disclaimer1 = ""
                        
                        # Color coding based on rating - 8 granular categories
                        if rating1 >= 0.9:
                            color1 = "#006400"
                            bg_color1 = "#d4edda"
                            badge1 = "ELITE"
                        elif rating1 >= 0.8:
                            color1 = "#228b22"
                            bg_color1 = "#d4edda"
                            badge1 = "GREAT"
                        elif rating1 >= 0.7:
                            color1 = "#32cd32"
                            bg_color1 = "#e8f5e8"
                            badge1 = "VERY GOOD"
                        elif rating1 >= 0.6:
                            color1 = "#9acd32"
                            bg_color1 = "#f0f8e8"
                            badge1 = "GOOD"
                        elif rating1 >= 0.5:
                            color1 = "#ff7f0e"
                            bg_color1 = "#fff3cd"
                            badge1 = "ABOVE AVERAGE"
                        elif rating1 >= 0.4:
                            color1 = "#ffa500"
                            bg_color1 = "#fff8dc"
                            badge1 = "AVERAGE"
                        elif rating1 >= 0.3:
                            color1 = "#ff6347"
                            bg_color1 = "#ffe4e1"
                            badge1 = "BELOW AVERAGE"
                        else:
                            color1 = "#d62728"
                            bg_color1 = "#f8d7da"
                            badge1 = "POOR"
                        
                        
                        st.markdown(f"""
                        <div style='background-color: {bg_color1}; padding: 15px; border-radius: 10px; text-align: center; margin: 15px 0; border: 2px solid {color1};'>
                            <h2 style='color: {color1}; margin: 10px 0; font-size: 2rem;'>{rating1:.3f}</h2>
                            <p style='color: {color1}; margin: 5px 0; font-weight: bold;'>RATING</p>
                            <span style='background-color: {color1}; color: white; padding: 5px 10px; border-radius: 15px; font-size: 0.8rem; font-weight: bold;'>{badge1}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if disclaimer1:
                            st.markdown(f"<p style='text-align: center; color: #888; font-size: 0.9rem; font-style: italic;'>{disclaimer1}</p>", unsafe_allow_html=True)
            
            # Player 2 column
            with col2:
                st.markdown(f"### {player2['Name']}")
                
                # Basic info
                st.write(f"**Team:** {player2.get('Team', 'Unknown')}")
                st.write(f"**Draft Pick:** #{int(player2.get('Pick', 0)) if player2.get('Pick', 0) > 0 else 'Undrafted'}")
                st.write(f"**Height:** {player2.get('Height', 'Unknown')}")
                
                # Key stats
                st.write("**Key Stats:**")
                st.write(f"**BPM:** {player2.get('BPM', 0):.1f}")
                st.write(f"**OREB%+DREB%:** {player2.get('REB', 0):.1f}")
                st.write(f"**Ast%:** {player2.get('Ast', 0):.1f}")
                st.write(f"**Blk%:** {player2.get('Blk', 0):.1f}")
                
                # Calculate prediction score
                cluster2 = player2["PlayStyleCluster"]
                if cluster2 in models_by_cluster:
                    # Get 2019-2025 players in same cluster for comparison
                    cluster_players = final_df_transform[
                        (final_df_transform['PlayStyleCluster'] == cluster2) & 
                        (final_df_transform['Year'] >= 10) & 
                        (final_df_transform['Year'] <= 25)
                    ]
                    
                    if len(cluster_players) > 0:
                        # Calculate predictions for all players in cluster
                        model_data = models_by_cluster[cluster2]
                        features = model_data["features"]
                        scaler = model_data["scaler"]
                        avg_coefs = model_data["avg_coefs"]
                        
                        # Get player2's prediction
                        player2_features = player2[features].fillna(0)
                        player2_scaled = scaler.transform(player2_features.values.reshape(1, -1))
                        logit2 = np.dot(player2_scaled, avg_coefs)
                        prob2 = 1 / (1 + np.exp(-logit2))
                        
                        
                        # Calculate all cluster predictions for ranking
                        predictions = []
                        for idx, row in cluster_players.iterrows():
                            try:
                                row_features = row[features].fillna(0)
                                row_scaled = scaler.transform(row_features.values.reshape(1, -1))
                                row_logit = np.dot(row_scaled, avg_coefs)
                                row_prob = 1 / (1 + np.exp(-row_logit))
                                
                                predictions.append(row_prob[0])
                            except:
                                predictions.append(0)
                        
                        cluster_players = cluster_players.copy()
                        cluster_players['Prediction'] = predictions
                        cluster_players = cluster_players.sort_values('Prediction', ascending=False).reset_index(drop=True)
                        
                        # Check if player2 is in 2019-2025
                        player2_year = player2.get('Year', 0)
                        if 19 <= player2_year <= 24:
                            # Player is in 2019-2025, show actual rank
                            player2_rank_idx = cluster_players[cluster_players['Name'] == player2['Name']].index
                            if len(player2_rank_idx) > 0:
                                rank2 = player2_rank_idx[0] + 1
                                total2 = len(cluster_players)
                        # Calculate player2's rating (prediction score)
                        player2_year = player2.get('Year', 0)
                        rating2 = prob2[0]
                        disclaimer2 = ""
                        
                        # Color coding based on rating - 8 granular categories
                        if rating2 >= 0.9:
                            color2 = "#006400"
                            bg_color2 = "#d4edda"
                            badge2 = "ELITE"
                        elif rating2 >= 0.8:
                            color2 = "#228b22"
                            bg_color2 = "#d4edda"
                            badge2 = "GREAT"
                        elif rating2 >= 0.7:
                            color2 = "#32cd32"
                            bg_color2 = "#e8f5e8"
                            badge2 = "VERY GOOD"
                        elif rating2 >= 0.6:
                            color2 = "#9acd32"
                            bg_color2 = "#f0f8e8"
                            badge2 = "GOOD"
                        elif rating2 >= 0.5:
                            color2 = "#ff7f0e"
                            bg_color2 = "#fff3cd"
                            badge2 = "ABOVE AVERAGE"
                        elif rating2 >= 0.4:
                            color2 = "#ffa500"
                            bg_color2 = "#fff8dc"
                            badge2 = "AVERAGE"
                        elif rating2 >= 0.3:
                            color2 = "#ff6347"
                            bg_color2 = "#ffe4e1"
                            badge2 = "BELOW AVERAGE"
                        else:
                            color2 = "#d62728"
                            bg_color2 = "#f8d7da"
                            badge2 = "POOR"
                        
                        
                        st.markdown(f"""
                        <div style='background-color: {bg_color2}; padding: 15px; border-radius: 10px; text-align: center; margin: 15px 0; border: 2px solid {color2};'>
                            <h2 style='color: {color2}; margin: 10px 0; font-size: 2rem;'>{rating2:.3f}</h2>
                            <p style='color: {color2}; margin: 5px 0; font-weight: bold;'>RATING</p>
                            <span style='background-color: {color2}; color: white; padding: 5px 10px; border-radius: 15px; font-size: 0.8rem; font-weight: bold;'>{badge2}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if disclaimer2:
                            st.markdown(f"<p style='text-align: center; color: #888; font-size: 0.9rem; font-style: italic;'>{disclaimer2}</p>", unsafe_allow_html=True)

with tab4:
    # Header with styling
    st.markdown("""
    <div style='text-align: center; padding: 15px; background: linear-gradient(90deg, #9932cc, #8a2be2); border-radius: 10px; margin-bottom: 20px;'>
        <h1 style='color: white; margin: 0;'>Rankings & Analysis</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Add explanation
    st.markdown("""
    <div style='background-color: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107; margin-bottom: 20px;'>
        <strong style='color: #856404;'>Rankings Explanation:</strong> The rankings below show only players from 2019-2025, ranked by their ratings.
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Lottery Picks Analysis (2019-2025)")
    st.write("*Players drafted in picks 1-14, ranked by rating*")
    
    # Add model training info
    st.markdown("""
    <div style='background-color: #e7f3ff; padding: 15px; border-radius: 8px; border-left: 4px solid #1f77b4; margin-bottom: 20px;'>
        <strong style='color: #1f77b4;'>Model Training:</strong> This model was trained on data from 2010-2018
    </div>
    """, unsafe_allow_html=True)
    
    try:
        # Get all predictions for years 2019-2025 using the same logic as show_top_predictions_by_year
        combined_rows = []
        
        # Loop through years 2019-2025 for rankings display
        for year in range(19, 26):  # 19, 20, 21, 22, 23, 24, 25
            for cluster_id, model_data in models_by_cluster.items():
                # Filter by year and cluster
                year_cluster_players = final_df_transform[
                    (final_df_transform['Year'] == year) & 
                    (final_df_transform['PlayStyleCluster'] == cluster_id)
                ].copy()
                
                if year_cluster_players.empty:
                    continue
                
                # Calculate predictions for all players in this cluster/year
                cluster_features = model_data["features"]
                cluster_scaler = model_data["scaler"]
                cluster_coefs = model_data["avg_coefs"]
                
                predictions = []
                for idx, row in year_cluster_players.iterrows():
                    row_features = row[cluster_features].fillna(0)
                    row_scaled = cluster_scaler.transform(row_features.values.reshape(1, -1))
                    row_logit = np.dot(row_scaled, cluster_coefs)
                    row_prob = 1 / (1 + np.exp(-row_logit))
                    
                    predictions.append(row_prob[0])
                
                year_cluster_players['Prediction'] = predictions
                year_cluster_players['Cluster'] = cluster_id
                combined_rows.append(year_cluster_players[['Name', 'Year', 'Prediction', 'Cluster', 'Team', 'Pick']])
        
        if combined_rows:
            all_preds = pd.concat(combined_rows)
            
            # Get lottery picks and their prediction scores
            lottery_picks_with_scores = []
            
            for cluster_id in [0.0, 1.0, 2.0]:
                # Get all 2019-2025 players in this position for display
                display_players = all_preds[all_preds['Cluster'] == cluster_id].copy()
                
                # Get lottery picks for this position (from 2019-2025 players)
                position_lottery = display_players[(display_players['Pick'] <= 14) & (display_players['Pick'].notna())].copy()
                
                # Get all 2010-2025 players in this position for rating calculation
                comparison_players = final_df_transform[
                    (final_df_transform['PlayStyleCluster'] == cluster_id) & 
                    (final_df_transform['Year'] >= 10) & 
                    (final_df_transform['Year'] <= 25)
                ].copy()
                
                if len(comparison_players) > 0 and cluster_id in models_by_cluster:
                    model_data = models_by_cluster[cluster_id]
                    features = model_data["features"]
                    scaler = model_data["scaler"]
                    avg_coefs = model_data["avg_coefs"]
                    
                    # Calculate predictions for all 2019-2025 players for rating ranking
                    comparison_predictions = []
                    for idx, row in comparison_players.iterrows():
                        row_features = row[features].fillna(0)
                        row_scaled = scaler.transform(row_features.values.reshape(1, -1))
                        row_logit = np.dot(row_scaled, avg_coefs)
                        row_prob = 1 / (1 + np.exp(-row_logit))
                        
                        comparison_predictions.append(row_prob[0])
                    
                    comparison_players['Prediction'] = comparison_predictions
                    comparison_players = comparison_players.sort_values('Prediction', ascending=False).reset_index(drop=True)
                    
                    # Add lottery picks with their ratings within position
                    for idx, player in position_lottery.iterrows():
                        # Find player's rank within their position
                        player_rank_idx = comparison_players[comparison_players['Name'] == player['Name']].index
                        if len(player_rank_idx) > 0:
                            rank = player_rank_idx[0] + 1  # 1-based rank
                            total = len(comparison_players)
                            # Use prediction score as rating
                            rating = player['Prediction']
                        else:
                            # Fallback calculation
                            player_prediction = player['Prediction']
                            lower_preds = len([p for p in comparison_predictions if p < player_prediction])
                            total = len(comparison_predictions)
                            rating = player['Prediction']
                        
                        player_data = player.copy()
                        player_data['Rating'] = rating
                        lottery_picks_with_scores.append(player_data)
            
            if lottery_picks_with_scores:
                # Convert to DataFrame and sort by rating
                lottery_df = pd.DataFrame(lottery_picks_with_scores)
                lottery_df = lottery_df.sort_values('Rating', ascending=False)
                
                # Show all lottery picks ranked by rating
                st.subheader("All Lottery Picks Ranked by Rating")
                
                for i, (_, player) in enumerate(lottery_df.iterrows(), 1):
                    year_display = 2000 + player['Year']
                    pick_num = int(player['Pick']) if not pd.isna(player['Pick']) else 'Unknown'
                    rating = player['Rating']
                    
                    st.markdown(f"**{i}. {player['Name']}** - Pick #{pick_num}")
                    st.write(f"   {player.get('Team', 'Unknown')}, {year_display}")
                    st.write(f"   Rating: {rating:.3f}")
                    st.write("")
            else:
                st.write("No lottery picks found with rating data")
        
        else:
            st.write("No data available for this analysis")
            
    except Exception as e:
        st.error(f"Error calculating top lottery picks: {str(e)}")

    st.markdown("---")
    
    # Add steals section
    st.subheader("Draft Steals Analysis (2019-2025)")
    st.write("*High rating players drafted after the lottery (picks 15+)*")
    
    try:
        if combined_rows:
            all_preds = pd.concat(combined_rows)
            
            # Get draft steals and their prediction scores
            steals_with_scores = []
            
            for cluster_id in [0.0, 1.0, 2.0]:
                # Get all 2019-2025 players in this position for display
                display_players = all_preds[all_preds['Cluster'] == cluster_id].copy()
                
                # Get non-lottery picks for this position (Pick > 14) from 2019-2025 players
                position_steals = display_players[(display_players['Pick'] > 14) & (display_players['Pick'].notna())].copy()
                
                # Get all 2010-2025 players in this position for rating calculation
                comparison_players = final_df_transform[
                    (final_df_transform['PlayStyleCluster'] == cluster_id) & 
                    (final_df_transform['Year'] >= 10) & 
                    (final_df_transform['Year'] <= 25)
                ].copy()
                
                if len(comparison_players) > 0 and cluster_id in models_by_cluster:
                    model_data = models_by_cluster[cluster_id]
                    features = model_data["features"]
                    scaler = model_data["scaler"]
                    avg_coefs = model_data["avg_coefs"]
                    
                    # Calculate predictions for all 2019-2025 players for rating ranking
                    comparison_predictions = []
                    for idx, row in comparison_players.iterrows():
                        row_features = row[features].fillna(0)
                        row_scaled = scaler.transform(row_features.values.reshape(1, -1))
                        row_logit = np.dot(row_scaled, avg_coefs)
                        row_prob = 1 / (1 + np.exp(-row_logit))
                        
                        comparison_predictions.append(row_prob[0])
                    
                    comparison_players['Prediction'] = comparison_predictions
                    comparison_players = comparison_players.sort_values('Prediction', ascending=False).reset_index(drop=True)
                    
                    # Add non-lottery picks with good ratings within position
                    for idx, player in position_steals.iterrows():
                        # Find player's rank within their position
                        player_rank_idx = comparison_players[comparison_players['Name'] == player['Name']].index
                        if len(player_rank_idx) > 0:
                            rank = player_rank_idx[0] + 1  # 1-based rank
                            total = len(comparison_players)
                            # Use prediction score as rating
                            rating = player['Prediction']
                        else:
                            # Fallback calculation
                            player_prediction = player['Prediction']
                            lower_preds = len([p for p in comparison_predictions if p < player_prediction])
                            total = len(comparison_predictions)
                            rating = player['Prediction']
                        
                        # Only include players with decent ratings (0.3+)
                        if rating >= 0.3:
                            player_data = player.copy()
                            player_data['Rating'] = rating
                            steals_with_scores.append(player_data)
            
            if steals_with_scores:
                # Convert to DataFrame and sort by rating
                steals_df = pd.DataFrame(steals_with_scores)
                steals_df = steals_df.sort_values('Rating', ascending=False)
                
                # Display top 25 steals
                st.subheader("Top 25 Draft Steals by Rating")
                top_steals = steals_df.head(25)
                
                for i, (_, player) in enumerate(top_steals.iterrows(), 1):
                    year_display = 2000 + player['Year']
                    pick_num = int(player['Pick']) if not pd.isna(player['Pick']) else 'Unknown'
                    rating = player['Rating']
                    
                    st.markdown(f"**{i}. {player['Name']}** - Pick #{pick_num}")
                    st.write(f"   {player.get('Team', 'Unknown')}, {year_display}")
                    st.write(f"   Rating: {rating:.3f}")
                    st.write("")
            else:
                st.write("No draft steals found with good ratings")
        
        else:
            st.write("No data available for this analysis")
            
    except Exception as e:
        st.error(f"Error calculating draft steals: {str(e)}")

    st.markdown("---")

with tab5:
    # Header with styling
    st.markdown("""
    <div style='text-align: center; padding: 15px; background: linear-gradient(90deg, #6f42c1, #e83e8c); border-radius: 10px; margin-bottom: 20px;'>
        <h1 style='color: white; margin: 0;'>Draft Class Analysis by Year</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("*Enter a year to see all drafted players ordered by draft pick with their prediction scores*")
    
    # Add model training info
    st.markdown("""
    <div style='background-color: #e7f3ff; padding: 15px; border-radius: 8px; border-left: 4px solid #1f77b4; margin-bottom: 20px;'>
        <strong style='color: #1f77b4;'>Note:</strong> Only players from 2019-2025 are available for analysis. Players are ranked by their ratings.
    </div>
    """, unsafe_allow_html=True)
    
    # Year input
    selected_year = st.number_input("Enter draft year (e.g., 2019, 2020, etc.):", 
                                   min_value=2019, max_value=2025, value=2019, step=1)
    
    # Convert to internal year format (subtract 2000)
    internal_year = selected_year - 2000
    
    try:
        # Get all players from the selected year
        year_players = final_df_transform[
            (final_df_transform['Year'] == internal_year) & 
            (final_df_transform['Pick'].notna()) & 
            (final_df_transform['Pick'] > 0)
        ].copy()
        
        if len(year_players) > 0 and 2019 <= selected_year <= 2025:
            # Calculate predictions for all players
            year_players_with_ratings = []
            
            for idx, player in year_players.iterrows():
                cluster = player["PlayStyleCluster"]
                if cluster in models_by_cluster:
                    model_data = models_by_cluster[cluster]
                    features = model_data["features"]
                    scaler = model_data["scaler"]
                    avg_coefs = model_data["avg_coefs"]
                    
                    # Calculate prediction
                    player_features = player[features].fillna(0)
                    player_scaled = scaler.transform(player_features.values.reshape(1, -1))
                    logit = np.dot(player_scaled, avg_coefs)
                    prob = 1 / (1 + np.exp(-logit))
                    
                    
                    # Get comparison players for rating calculation
                    if 19 <= internal_year <= 25:
                        # Use 2010-2025 for rating calculation
                        comparison_players = final_df_transform[
                            (final_df_transform['PlayStyleCluster'] == cluster) & 
                            (final_df_transform['Year'] >= 10) & 
                            (final_df_transform['Year'] <= 25)
                        ]
                    else:
                        # For 2010-2018, compare against full 2010-2025 dataset
                        comparison_players = final_df_transform[
                            (final_df_transform['PlayStyleCluster'] == cluster) & 
                            (final_df_transform['Year'] >= 10) & 
                            (final_df_transform['Year'] <= 25)
                        ]
                    
                    if len(comparison_players) > 0:
                        # Calculate predictions for comparison players
                        comparison_predictions = []
                        for _, comp_player in comparison_players.iterrows():
                            try:
                                comp_features = comp_player[features].fillna(0)
                                comp_scaled = scaler.transform(comp_features.values.reshape(1, -1))
                                comp_logit = np.dot(comp_scaled, avg_coefs)
                                comp_prob = 1 / (1 + np.exp(-comp_logit))
                                
                                comparison_predictions.append(comp_prob[0])
                            except:
                                comparison_predictions.append(0)
                        
                        # Calculate rating within position
                        comparison_players = comparison_players.copy()
                        comparison_players['Prediction'] = comparison_predictions
                        comparison_players = comparison_players.sort_values('Prediction', ascending=False).reset_index(drop=True)
                        
                        if 19 <= internal_year <= 25:
                            # Actual rating for 2019-2025 players within their position
                            player_rank_idx = comparison_players[comparison_players['Name'] == player['Name']].index
                            if len(player_rank_idx) > 0:
                                rank = player_rank_idx[0] + 1  # 1-based rank
                                total = len(comparison_players)
                                # Use prediction score as rating
                                rating = prob[0]
                                is_hypothetical = False
                            else:
                                rating = 0
                                is_hypothetical = False
                        else:
                            # Hypothetical rating for 2010-2018 players within position
                            lower_preds = len([p for p in comparison_predictions if p < prob[0]])
                            total = len(comparison_predictions)
                            rating = prob[0]
                            is_hypothetical = True
                        
                        # Add player data
                        player_data = {
                            'Name': player['Name'],
                            'Pick': int(player['Pick']),
                            'Team': player.get('Team', 'Unknown'),
                            'Rating': rating,
                            'Prediction': prob[0],
                            'IsHypothetical': is_hypothetical
                        }
                        year_players_with_ratings.append(player_data)
            
            if year_players_with_ratings:
                # Convert to DataFrame and sort by draft pick
                year_df = pd.DataFrame(year_players_with_ratings)
                year_df = year_df.sort_values('Pick')
                
                st.subheader(f"{selected_year} Draft Class ({len(year_df)} players)")
                
                # Show players
                for i, (_, player) in enumerate(year_df.iterrows(), 1):
                    rating = player['Rating']
                    hypothetical_text = " (Hypothetical)" if player['IsHypothetical'] else ""
                    
                    # Color coding based on rating - 8 granular categories
                    if rating >= 0.9:
                        color = "#006400"
                        badge = "ELITE"
                    elif rating >= 0.8:
                        color = "#228b22"
                        badge = "GREAT"
                    elif rating >= 0.7:
                        color = "#32cd32"
                        badge = "VERY GOOD"
                    elif rating >= 0.6:
                        color = "#9acd32"
                        badge = "GOOD"
                    elif rating >= 0.5:
                        color = "#ff7f0e"
                        badge = "ABOVE AVERAGE"
                    elif rating >= 0.4:
                        color = "#ffa500"
                        badge = "AVERAGE"
                    elif rating >= 0.3:
                        color = "#ff6347"
                        badge = "BELOW AVERAGE"
                    else:
                        color = "#d62728"
                        badge = "POOR"
                    
                    st.markdown(f"""
                    <div style='background-color: #f8f9fa; padding: 12px; border-radius: 8px; margin-bottom: 8px; border-left: 4px solid {color};'>
                        <strong>Pick #{player['Pick']}: {player['Name']}</strong> - {player['Team']}<br>
                        <span style='color: {color}; font-weight: bold;'>{rating:.3f} rating{hypothetical_text}</span> | <span style='background-color: {color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem;'>{badge}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("No players found with rating data for this year")
        else:
            if selected_year < 2019:
                st.warning(f"Data for {selected_year} is not available. Only years 2019-2025 are supported.")
            else:
                st.write(f"No drafted players found for {selected_year}")
            
    except Exception as e:
        st.error(f"Error analyzing {selected_year} draft class: {str(e)}")

with tab6:
    # Header with styling
    st.markdown("""
    <div style='text-align: center; padding: 15px; background: linear-gradient(90deg, #6610f2, #fd7e14); border-radius: 10px; margin-bottom: 20px;'>
        <h1 style='color: white; margin: 0;'>Player Rankings by Rating</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("View all players ranked by their rating values for a specific year")
    
    # Add model training info
    st.markdown("""
    <div style='background-color: #e7f3ff; padding: 15px; border-radius: 8px; border-left: 4px solid #1f77b4; margin-bottom: 20px;'>
        <strong style='color: #1f77b4;'>Note:</strong> Players are ranked by their rating values. Only players from 2019-2025 are available for ranking.
    </div>
    """, unsafe_allow_html=True)
    
    # Year selection
    ranking_year = st.selectbox("Select Year for Rankings:", 
                               options=list(range(2019, 2026)), 
                               index=0, 
                               key="ranking_year")
    
    # Number of players to show
    num_players = st.slider("Number of players to show:", min_value=10, max_value=100, value=50, step=10)
    
    # Convert year to internal format
    internal_year = ranking_year - 2000
    cluster_filter = [0.0, 1.0, 2.0]  # Include all play styles
    
    try:
        # Get all players from the selected year and positions
        year_players = final_df_transform[
            (final_df_transform['Year'] == internal_year) & 
            (final_df_transform['PlayStyleCluster'].isin(cluster_filter))
        ].copy()
        
        if len(year_players) > 0:
            # Calculate ratings for each player
            all_players_with_ratings = []
            
            for idx, player in year_players.iterrows():
                cluster = player["PlayStyleCluster"]
                if cluster in models_by_cluster:
                    model_data = models_by_cluster[cluster]
                    features = model_data["features"]
                    scaler = model_data["scaler"]
                    avg_coefs = model_data["avg_coefs"]
                    
                    # Calculate prediction for this player
                    player_features = player[features].fillna(0)
                    player_scaled = scaler.transform(player_features.values.reshape(1, -1))
                    logit = np.dot(player_scaled, avg_coefs)
                    prob = 1 / (1 + np.exp(-logit))
                    
                    # Get all players in same position for rating calculation (2010-2025)
                    comparison_players = final_df_transform[
                        (final_df_transform['PlayStyleCluster'] == cluster) & 
                        (final_df_transform['Year'] >= 10) & 
                        (final_df_transform['Year'] <= 25)
                    ]
                    
                    if len(comparison_players) > 0:
                        # Calculate predictions for all comparison players
                        comparison_predictions = []
                        for _, comp_player in comparison_players.iterrows():
                            try:
                                comp_features = comp_player[features].fillna(0)
                                comp_scaled = scaler.transform(comp_features.values.reshape(1, -1))
                                comp_logit = np.dot(comp_scaled, avg_coefs)
                                comp_prob = 1 / (1 + np.exp(-comp_logit))
                                comparison_predictions.append(comp_prob[0])
                            except:
                                comparison_predictions.append(0)
                        
                        # Calculate rating within position
                        comparison_players = comparison_players.copy()
                        comparison_players['Prediction'] = comparison_predictions
                        comparison_players = comparison_players.sort_values('Prediction', ascending=False).reset_index(drop=True)
                        
                        # Find player's rank within their position
                        player_rank_idx = comparison_players[comparison_players['Name'] == player['Name']].index
                        if len(player_rank_idx) > 0:
                            rank = player_rank_idx[0] + 1  # 1-based rank
                            total = len(comparison_players)
                            rating = prob[0]
                        else:
                            # Fallback calculation if player not found
                            lower_preds = len([p for p in comparison_predictions if p < prob[0]])
                            total = len(comparison_predictions)
                            rating = prob[0]
                        
                        player_data = {
                            'Name': player['Name'],
                            'Team': player.get('Team', 'Unknown'),
                            'Rating': rating,
                            'Prediction': prob[0],
                            'Pick': player.get('Pick', None),
                            'Height': player.get('Height', 'Unknown'),
                            'BPM': player.get('BPM', 0),
                            'Ast': player.get('Ast', 0),
                            'REB': player.get('REB', 0),
                            'Blk': player.get('Blk', 0)
                        }
                        all_players_with_ratings.append(player_data)
            
            if all_players_with_ratings:
                # Convert to DataFrame and sort by rating
                rankings_df = pd.DataFrame(all_players_with_ratings)
                rankings_df = rankings_df.sort_values('Rating', ascending=False)
                
                # Limit to requested number of players
                top_players = rankings_df.head(num_players)
                
                st.subheader(f"Top {len(top_players)} Players from {ranking_year} (Ranked by Rating)")
                
                # Display players in a nice format
                for i, (_, player) in enumerate(top_players.iterrows(), 1):
                    rating = player['Rating']
                    pick_text = f"Pick #{int(player['Pick'])}" if pd.notna(player['Pick']) and player['Pick'] > 0 else "Undrafted"
                    
                    # Color coding based on rating - 8 granular categories
                    if rating >= 0.9:
                        color = "#006400"
                        bg_color = "#d4edda"
                        badge = "ELITE"
                    elif rating >= 0.8:
                        color = "#228b22"
                        bg_color = "#d4edda"
                        badge = "GREAT"
                    elif rating >= 0.7:
                        color = "#32cd32"
                        bg_color = "#e8f5e8"
                        badge = "VERY GOOD"
                    elif rating >= 0.6:
                        color = "#9acd32"
                        bg_color = "#f0f8e8"
                        badge = "GOOD"
                    elif rating >= 0.5:
                        color = "#ff7f0e"
                        bg_color = "#fff3cd"
                        badge = "ABOVE AVERAGE"
                    elif rating >= 0.4:
                        color = "#ffa500"
                        bg_color = "#fff8dc"
                        badge = "AVERAGE"
                    elif rating >= 0.3:
                        color = "#ff6347"
                        bg_color = "#ffe4e1"
                        badge = "BELOW AVERAGE"
                    else:
                        color = "#d62728"
                        bg_color = "#f8d7da"
                        badge = "POOR"
                    
                    st.markdown(f"""
                    <div style='background-color: {bg_color}; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 4px solid {color};'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div>
                                <strong style='font-size: 1.1rem;'>#{i}. {player['Name']}</strong> - {player['Team']}<br>
                                <span style='color: #666; font-size: 0.9rem;'>{player['Height']} | {pick_text}</span>
                            </div>
                            <div style='text-align: right;'>
                                <div style='color: {color}; font-size: 1.5rem; font-weight: bold;'>{rating:.3f}</div>
                                <span style='background-color: {color}; color: white; padding: 3px 8px; border-radius: 12px; font-size: 0.7rem; font-weight: bold;'>{badge}</span>
                            </div>
                        </div>
                        <div style='margin-top: 8px; font-size: 0.85rem; color: #555;'>
                            BPM: {player['BPM']:.1f} | Ast%: {player['Ast']:.1f} | REB%: {player['REB']:.1f} | Blk%: {player['Blk']:.1f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Add summary statistics
                st.markdown("---")
                st.subheader("Summary Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Average Rating", f"{top_players['Rating'].mean():.3f}")
                
                with col2:
                    exceptional_count = len(top_players[top_players['Rating'] >= 0.9])
                    st.metric("Superstar Players (0.9+)", exceptional_count)
                
                with col3:
                    elite_count = len(top_players[top_players['Rating'] >= 0.8])
                    st.metric("Elite Players (0.8+)", elite_count)
                
                
            else:
                st.write("No players found with rating data for the selected filters.")
        else:
            st.write(f"No players found for {ranking_year}.")
            
    except Exception as e:
        st.error(f"Error generating rankings: {str(e)}")


# Footer
st.markdown("---")
st.markdown("*Built with your NCAAB prediction models from the Jupyter notebook*")