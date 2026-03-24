from groq import Groq
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sqlite3
from datetime import datetime

def init_db():
    conn = sqlite3.connect("athlete_progress.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS progress 
                 (date TEXT, position TEXT, pace INT, shooting INT, passing INT, 
                 dribbling INT, defending INT, physic INT)''')
    conn.commit()
    conn.close()

def save_progress(stats, position):
    conn = sqlite3.connect("athlete_progress.db")
    c = conn.cursor()
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO progress VALUES (?, ?, ?, ?, ?, ?, ?, ?)", 
              (date_str, position, stats['pace'], stats['shooting'], stats['passing'], 
               stats['dribbling'], stats['defending'], stats['physic']))
    conn.commit()
    conn.close()

init_db()

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# --- 1. SETUP & PAGE CONFIG ---
st.set_page_config(page_title="KickSmart AI Scout", layout="wide")
st.title("⚽ KickSmart AI - Positional Recommender")
st.write("Enter an athlete's stats below to discover their statistically ideal position on the pitch.")

# --- 2. LOAD MODEL ASSETS ---
@st.cache_resource # Crucial: Caches the model so it doesn't reload on every UI interaction
def load_model():
    with open('kicksmart_model.pkl', 'rb') as file:
        return pickle.load(file)

try:
    assets = load_model()
    model = assets['model']
    final_features = assets['features']
    le_pos = assets['le_pos']
    le_work = assets['le_work']
except FileNotFoundError:
    st.error("Model file 'kicksmart_model.pkl' not found. Please run the training script first.")
    st.stop()

def get_ai_coaching_plan(position, stats):
    """
    Sends player data to Gemini and gets a custom training plan using the new SDK.
    """
    # 1. Identify strengths and weaknesses (same logic as before)
    skill_stats = {k: v for k, v in stats.items() 
                   if k not in ['height_cm', 'weight_kg', 'preferred_foot', 'work_rate', 'weak_foot']}
    
    # Handle edge case where stats might be empty or zero
    if not skill_stats:
        return "Error: No skill stats found to analyze."
        
    weakness = min(skill_stats, key=skill_stats.get)
    strength = max(skill_stats, key=skill_stats.get)

    # 2. Construct the Prompt
    prompt = f"""
    Act as a world-class football coach (Premier League level). 
    I have a player identified as a **{position}**.
    
    Here is their physical profile:
    - **Height/Weight:** {stats['height_cm']}cm, {stats['weight_kg']}kg
    - **Best Attribute:** {strength} ({stats[strength]}/99)
    - **Weakest Attribute:** {weakness} ({stats[weakness]}/99)
    - **Pace:** {stats['pace']}
    - **Shooting:** {stats['shooting']}
    - **Passing:** {stats['passing']}
    - **Dribbling:** {stats['dribbling']}
    - **Defending:** {stats['defending']}
    - **Physical:** {stats['physic']}

    Create a specialized **30-Day Training Plan** for this player.
    1. Focus heavily on improving their **{weakness}** while utilizing their **{strength}**.
    2. Include specific drills (with sets and reps).
    3. Keep the tone encouraging but professional.
    4. Format the output with clear headings (Day 1, Day 2, Day 3..., Day 30).
    """

    # 3. Call the Model using the NEW client syntax
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile"
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error connecting to AI Coach: {str(e)}"

# --- 3. BUILD THE USER INTERFACE ---
tab1, tab2, tab3 = st.tabs(["📊 Stat Input & Prediction", "🧠 Coach Chatbot", "📈 Progress Dashboard"])

with tab1:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("🏃‍♂️ Physical Profile")
        height_cm = st.number_input("Height (cm)", min_value=150, max_value=220, value=180)
        weight_kg = st.number_input("Weight (kg)", min_value=50, max_value=120, value=75)
        preferred_foot = st.selectbox("Preferred Foot", ["Right", "Left"])
        weak_foot = st.slider("Weak Foot (1-5)", 1, 5, 3)
        skill_moves = st.slider("Skill Moves (1-5)", 1, 5, 3)
        # These must exactly match the formats the model saw during training
        work_rate = st.selectbox("Work Rate (Att/Def)", 
            ["High/High", "High/Medium", "High/Low", 
             "Medium/High", "Medium/Medium", "Medium/Low", 
             "Low/High", "Low/Medium", "Low/Low"], index=4)

    with col2:
        st.header("⚡ Core Attributes")
        pace = st.slider("Pace", 1, 99, 75)
        shooting = st.slider("Shooting", 1, 99, 60)
        passing = st.slider("Passing", 1, 99, 70)
        dribbling = st.slider("Dribbling", 1, 99, 72)
        defending = st.slider("Defending", 1, 99, 65)
        physic = st.slider("Physicality", 1, 99, 70)
    
        st.subheader("Granular Movement")
        movement_acceleration = st.slider("Acceleration", 1, 99, 75)
        movement_sprint_speed = st.slider("Sprint Speed", 1, 99, 75)
    
    with col3:
        st.header("🛡️ Specialized Stats")
        power_stamina = st.slider("Stamina", 1, 99, 75)
        power_jumping = st.slider("Jumping", 1, 99, 70)
        power_strength = st.slider("Strength", 1, 99, 70)
        defending_standing_tackle = st.slider("Standing Tackle", 1, 99, 60)
        defending_sliding_tackle = st.slider("Sliding Tackle", 1, 99, 60)
    
        st.subheader("🧤 Goalkeeping")
        goalkeeping_diving = st.slider("GK Diving", 1, 99, 10)
        goalkeeping_handling = st.slider("GK Handling", 1, 99, 10)
        goalkeeping_kicking = st.slider("GK Kicking", 1, 99, 10)

    # --- 4. PREDICTION LOGIC ---
    if st.button("Predict Ideal Position", type="primary", use_container_width=True):
    
        # Gather all inputs into a dictionary
        player_stats = {
            'height_cm': height_cm, 'weight_kg': weight_kg, 
            'preferred_foot': preferred_foot, 'weak_foot': weak_foot, 
            'skill_moves': skill_moves, 'work_rate': work_rate, 
            'pace': pace, 'shooting': shooting, 'passing': passing, 
            'dribbling': dribbling, 'defending': defending, 'physic': physic,
            'movement_acceleration': movement_acceleration, 
            'movement_sprint_speed': movement_sprint_speed, 
            'power_stamina': power_stamina, 'power_jumping': power_jumping, 
            'power_strength': power_strength, 
            'defending_standing_tackle': defending_standing_tackle, 
            'defending_sliding_tackle': defending_sliding_tackle,
            'goalkeeping_diving': goalkeeping_diving, 
            'goalkeeping_handling': goalkeeping_handling, 
            'goalkeeping_kicking': goalkeeping_kicking
        }
    
        # Mirror the exact preprocessing pipeline from the training script
        input_df = pd.DataFrame([player_stats])
        input_df['work_rate'] = le_work.transform(input_df['work_rate'])
     
        for col in final_features:
            if 'preferred_foot_' in col:
                foot_val = col.replace('preferred_foot_', '')
                input_df[col] = (input_df['preferred_foot'] == foot_val).astype(int)
            
        if 'preferred_foot' in input_df.columns:
            input_df = input_df.drop('preferred_foot', axis=1)
        
        input_df = input_df[final_features]
    
        # Get predictions
        probabilities = model.predict_proba(input_df)[0]
        top_3_indices = np.argsort(probabilities)[::-1][:3]
        top_position_idx = top_3_indices[0]
    
        # SAVE TO SESSION STATE (This gives Streamlit a memory)
        st.session_state['prediction_made'] = True
        st.session_state['top_3_indices'] = top_3_indices
        st.session_state['probabilities'] = probabilities
        st.session_state['top_position'] = le_pos.inverse_transform([top_position_idx])[0]
        st.session_state['player_stats'] = player_stats
        save_progress(player_stats, st.session_state['top_position'])
       
    # --- 5. DISPLAY RESULTS & AI COACH ---
    # We check if a prediction exists in memory. If so, display the UI.
    if st.session_state.get('prediction_made', False):
    
        st.divider()
        st.subheader("🎯 KickSmart AI Recommendation")
    
        res_cols = st.columns(3)
        for i, idx in enumerate(st.session_state['top_3_indices']):
            pos_name = le_pos.inverse_transform([idx])[0]
            match_percentage = round(st.session_state['probabilities'][idx] * 100, 1)
        
            with res_cols[i]:
                st.metric(label=f"Choice #{i+1}", value=pos_name, delta=f"{match_percentage}% Match", delta_color="normal")

        st.divider()
        st.subheader("🧠 Generative AI Coach")
        st.write(f"Generate a personalized training plan for a **{st.session_state['top_position']}** with your specific stats.")

        # Because this button is no longer nested inside the first button, it will work perfectly!
        if st.button("Generate Training Plan", type="primary"):
            with st.spinner("Analyzing your stats and writing drills..."):
            
                # Call the AI function using the data saved in session state
                coaching_plan = get_ai_coaching_plan(st.session_state['top_position'], st.session_state['player_stats'])
                st.session_state['training_plan'] = coaching_plan
                st.markdown("### 📋 Your Personalized Plan")
                st.markdown(coaching_plan)

# --- TAB 2: INTERACTIVE CHATBOT ---
with tab2:
    st.header("Ask Your Virtual Coach")
    
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if prompt := st.chat_input("Ask a follow-up question about your training..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Give the LLM context about the user's specific training plan
        system_context = "You are a professional football coach."
        if 'training_plan' in st.session_state:
            system_context += f" The user's current personalized plan is: {st.session_state['training_plan']}"
            
        api_messages = [{"role": "system", "content": system_context}] + st.session_state.messages
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=api_messages
        )
        
        reply = response.choices[0].message.content
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

# --- TAB 3: PROGRESS DASHBOARD ---
with tab3:
    st.header("Player Development Tracking")
    
    
    conn = sqlite3.connect("athlete_progress.db")
    df_progress = pd.read_sql_query("SELECT * FROM progress", conn)
    conn.close()
    
    if not df_progress.empty:
        df_progress['date'] = pd.to_datetime(df_progress['date'])
        df_progress = df_progress.set_index('date')
        
        st.subheader("Core Attributes Over Time")
        st.line_chart(df_progress[['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']])
        
        st.subheader("Positional History")
        st.dataframe(df_progress[['position']].sort_index(ascending=False))
    else:
        st.info("No data yet. Run a prediction in the first tab to initialize your dashboard!")