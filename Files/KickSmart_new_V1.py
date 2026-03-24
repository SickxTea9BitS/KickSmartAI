import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv("D:/male_players.csv/male_players.csv", low_memory=False)
print("Dataset loaded successfully.")


# 2. PREPROCESSING
# ---------------------------------------------------------
# Filter out empty player positions
df = df[df['player_positions'].notna()]

# Initialize the first listed postion as the best/primary position for the player
df['best_pos'] = df['player_positions'].apply(lambda x: x.split(',')[0].strip())

# Fill missing outfield stats with 0 so Goalkeepers aren't deleted
stats_to_fill_zero = [
    'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
    'movement_acceleration', 'movement_sprint_speed', 'power_stamina', 
    'power_jumping', 'power_strength', 'defending_standing_tackle', 
    'defending_sliding_tackle'
]

# Fill missing values for outfield stats
df[stats_to_fill_zero] = df[stats_to_fill_zero].fillna(0)

# Fill missing values for GK stats (in case outfield players have them as NaN)
gk_stats = ['goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking']
df[gk_stats] = df[gk_stats].fillna(0)

# Define the features (Inputs) to be analyzed for importance in determining the player's position.
features = [
    'height_cm', 'weight_kg', 'preferred_foot', 'weak_foot', 'skill_moves', 'work_rate', 
    'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
    'movement_acceleration', 'movement_sprint_speed', 'power_stamina', 'power_jumping', 
    'power_strength', 'defending_standing_tackle', 'defending_sliding_tackle', 
    'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking'
]

# Drop rows where these specific stats are missing
df_clean = df.dropna(subset=features + ['best_pos']).copy()

# 3. ANALYSIS: FEATURE IMPORTANCE (Random Forest)
# ---------------------------------------------------------

# One hot encoding for preferred foot
df_clean = pd.get_dummies(df_clean, columns=['preferred_foot'], drop_first=True)

# Dynamically update the features list to handle the new dummy columns
final_features = [f for f in features if f != 'preferred_foot']
final_features.extend([col for col in df_clean.columns if 'preferred_foot_' in col])

# Encode positions (ST=0, CB=1, etc.) and work rate so the model can understand them
le_pos = LabelEncoder()
le_work= LabelEncoder()
y = le_pos.fit_transform(df_clean['best_pos'])
df_clean['work_rate']=le_work.fit_transform(df_clean['work_rate'])
X = df_clean[final_features]

# Train a simple Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Get feature importance scores
importances = pd.Series(model.feature_importances_, index=final_features).sort_values(ascending=False)

# 4. VISUALIZATION
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.barplot(x=importances.values, y=importances.index, hue=importances.index, palette='viridis', legend=False)
plt.title('Which Stats Determine a Player\'s Position?')
plt.xlabel('Importance Score')
plt.ylabel('Attribute')
plt.show()

# 5. HEATMAP: Average Stats by Position
# ---------------------------------------------------------
# Group by position and calculate average stats
pos_stats = df_clean.groupby('best_pos')[final_features].mean()

plt.figure(figsize=(12, 8))
sns.heatmap(pos_stats, annot=True, cmap='coolwarm', fmt='.0f')
plt.title('Average Attributes by Position')
plt.show()

# 6. GET TOP 3 RECOMMENDED POSITIONS
# ---------------------------------------------------------
def predict_top_3_positions(player_stats, model, final_features, le_pos, le_work):
    """
    Takes raw stats, pre-processes them, and outputs the top 3 
    recommended positions with percentage match scores.
    """
    # 1. Convert to DataFrame and encode work_rate
    input_df = pd.DataFrame([player_stats])
    input_df['work_rate'] = le_work.transform(input_df['work_rate'])
    
    # 2. Handle preferred_foot dummy variables dynamically
    for col in final_features:
        if 'preferred_foot_' in col:
            foot_val = col.replace('preferred_foot_', '')
            input_df[col] = (input_df['preferred_foot'] == foot_val).astype(int)
            
    if 'preferred_foot' in input_df.columns:
        input_df = input_df.drop('preferred_foot', axis=1)
        
    # 3. Reorder columns to match training
    input_df = input_df[final_features]
    
    # 4. NEW: Get probabilities for all classes instead of a single prediction
    probabilities = model.predict_proba(input_df)[0]
    
    # 5. Get the indices of the top 3 probabilities in descending order
    top_3_indices = np.argsort(probabilities)[::-1][:3]
    
    # 6. Map indices back to position names and format the percentages
    results = []
    for idx in top_3_indices:
        position = le_pos.inverse_transform([idx])[0]
        match_percentage = round(probabilities[idx] * 100, 1)
        results.append((position, match_percentage))
        
    return results

# --- TEST IT OUT ---
# Our hypothetical defensive powerhouse with terrible shooting
test_athlete = {
    'height_cm': 180,
    'weight_kg': 75,
    'preferred_foot': 'Right',
    'weak_foot': 3,
    'skill_moves': 3,
    'work_rate': 'High/High', 
    'pace': 88,
    'shooting': 40,
    'passing': 72,
    'dribbling': 75,
    'defending': 82,
    'physic': 80,
    'movement_acceleration': 89,
    'movement_sprint_speed': 87,
    'power_stamina': 92,
    'power_jumping': 78,
    'power_strength': 76,
    'defending_standing_tackle': 84,
    'defending_sliding_tackle': 85,
    'goalkeeping_diving': 10,
    'goalkeeping_handling': 10,
    'goalkeeping_kicking': 10
}

# Run the prediction
top_positions = predict_top_3_positions(test_athlete, model, final_features, le_pos, le_work)

print("\n" + "="*50)
print("🏆 KICKSMART AI - POSITIONAL ANALYSIS 🏆")
print("="*50)
for i, (pos, prob) in enumerate(top_positions, 1):
    print(f"Rank {i}: {pos.ljust(5)} -> {prob}% Match")
print("="*50 + "\n")

# 7. EXPORT FOR WEB DEPLOYMENT
# ---------------------------------------------------------
print("\nSaving KickSmart AI model assets...")

# Package the model, the feature list, and both encoders together
kicksmart_assets = {
    'model': model,
    'features': final_features,
    'le_pos': le_pos,
    'le_work': le_work
}

# Write the package to a file
with open('kicksmart_model.pkl', 'wb') as file:
    pickle.dump(kicksmart_assets, file)

print("Success! 'kicksmart_model.pkl' is ready for deployment.")