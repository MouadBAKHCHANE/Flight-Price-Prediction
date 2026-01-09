import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
import kagglehub
from PIL import Image

# Settings
st.set_page_config(page_title="Indian Aviation Dashboard", layout="wide")

# --- Load Data & Model ---
@st.cache_resource
def load_resources():
    # Load Data
    if os.path.exists('Clean_Dataset.csv'):
        df = pd.read_csv('Clean_Dataset.csv')
    else:
        try:
            path = kagglehub.dataset_download("shubhambathwal/flight-price-prediction")
            file_path = os.path.join(path, 'Clean_Dataset.csv')
            df = pd.read_csv(file_path)
            if 'Unnamed: 0' in df.columns:
                df = df.drop(columns=['Unnamed: 0'])
        except Exception as e:
            st.error(f"Dataset not found and download failed: {e}")
            return None, None

    # Load Model
    try:
        model = joblib.load('flight_price_model.pkl')
    except:
        model = None
        
    return df, model

df_raw, pipeline = load_resources()

if df_raw is not None:
    df = df_raw.copy()
    # Feature Engineering for Analysis
    df['flight_date'] = pd.Timestamp('2022-02-11') + pd.to_timedelta(df['days_left'], unit='D')
    df['Month'] = df['flight_date'].dt.month
    
    # Dep_Hour map
    dep_time_map = {
        'Early_Morning': 5, 'Morning': 9, 'Afternoon': 14,
        'Evening': 19, 'Night': 22, 'Late_Night': 2
    }

import base64

def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- Sidebar (Profile & Filters) ---
with st.sidebar:
    # Custom HTML for centered profile with tight spacing
    if os.path.exists("profile_pic.png"):
        img_base64 = get_img_as_base64("profile_pic.png")
        img_html = f'<img src="data:image/png;base64,{img_base64}" style="border-radius: 50%; width: 120px; display: block; margin-left: auto; margin-right: auto;">'
    else:
        # Fallback logo
        img_html = '<img src="https://upload.wikimedia.org/wikipedia/en/thumb/9/9b/Indian_Airlines_Logo.svg/1200px-Indian_Airlines_Logo.svg.png" style="width: 120px; display: block; margin-left: auto; margin-right: auto;">'

    st.markdown(f"""
        <div style="text-align: center; margin-top: -20px;">
            {img_html}
            <h3 style="margin-top: 10px; margin-bottom: 0px;">Flight Price Prediction</h3>
        </div>
        <hr style="margin-top: 10px; margin-bottom: 20px;">
    """, unsafe_allow_html=True)

# --- Top Navigation ---
c_nav1, c_nav2, c_nav3 = st.columns([1, 6, 1])
with c_nav2:
    page = st.radio("Navigation", ["Home", "Dashboard", "Predict", "About Me"], horizontal=True, label_visibility="collapsed")

# --- Pages ---

if page == "Home":
    # st.title("✈️ Indian Aviation Flight Price Prediction") # Removed per user request
    
    # Display Airfare Image
    if os.path.exists("airfare.png"):
        # Use columns to reduce size and center the image
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image("airfare.png", use_container_width=True)
    elif os.path.exists("airfare.webp"):
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image("airfare.webp", use_container_width=True)
    else:
        st.info("Project Image Placeholder")
        
    st.markdown("### Project Overview")
    st.write("""
    This project aims to analyze the flight prices in the Indian Aviation Market and predict future prices based on various factors such as airline, journey duration, departure time, and stops.
    
    **Key Features:**
    *   **Interactive Dashboard**: Exploratory Data Analysis (EDA); Interactive visualizations to understand price trends.
    *   **Price Prediction**: A Machine Learning model (Gradient Boosting) to estimate fare.
    """)

elif page == "Dashboard":
    st.title("📊 Dashboard")
    
    if df is not None:
        # --- Filters in Sidebar ---
        with st.sidebar:
            # CSS to compact the selectboxes
            st.markdown("""
                <style>
                div[data-testid="stSidebar"] div[data-baseweb="select"] > div {
                    min-height: 35px;
                    padding-top: 0px;
                    padding-bottom: 0px;
                    font-size: 0.85rem;
                }
                div[data-testid="stSidebar"] label {
                    font-size: 0.85rem;
                    margin-bottom: 0px;
                    margin-top: 5px;
                }
                div[data-testid="stSidebar"] .stSelectbox {
                    margin-bottom: -15px;
                }
                </style>
            """, unsafe_allow_html=True)
            
            # Removed redundant "---" to move upper
            st.markdown("### Filter Data") # Changed header to markdown for better size control
            dept_city = st.selectbox("Departure City", options=['All'] + list(df['source_city'].unique()))
            arr_city = st.selectbox("Arrival City", options=['All'] + list(df['destination_city'].unique()))
            airline_filter = st.selectbox("Airline", options=['All'] + list(df['airline'].unique()))
            service_class = st.selectbox("Service Class", options=['All'] + list(df['class'].unique()))
            stops_filter = st.selectbox("Stops", options=['All'] + list(df['stops'].unique()))
            time_filter = st.selectbox("Departure Time", options=['All'] + list(df['departure_time'].unique()))

        # Apply Filters
        df_filtered = df.copy()
        if dept_city != 'All':
            df_filtered = df_filtered[df_filtered['source_city'] == dept_city]
        if arr_city != 'All':
            df_filtered = df_filtered[df_filtered['destination_city'] == arr_city]
        if airline_filter != 'All':
            df_filtered = df_filtered[df_filtered['airline'] == airline_filter]
        if service_class != 'All':
            df_filtered = df_filtered[df_filtered['class'] == service_class]
        if stops_filter != 'All':
            df_filtered = df_filtered[df_filtered['stops'] == stops_filter]
        if time_filter != 'All':
            df_filtered = df_filtered[df_filtered['departure_time'] == time_filter]

        # --- Dashboard ---
        # KPI Cards
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Flight Count", f"{len(df_filtered)}")
        k2.metric("Highest Price", f"₹ {df_filtered['price'].max():,.0f}" if not df_filtered.empty else "0")
        k3.metric("Lowest Price", f"₹ {df_filtered['price'].min():,.0f}" if not df_filtered.empty else "0")
        k4.metric("Top Airline", df_filtered['airline'].mode()[0] if not df_filtered.empty else "N/A")

        st.markdown("---")

        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Market Share")
            if not df_filtered.empty:
                airline_counts = df_filtered['airline'].value_counts().reset_index()
                airline_counts.columns = ['Airline', 'Count']
                fig_donut = px.pie(airline_counts, values='Count', names='Airline', hole=0.5)
                fig_donut.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_donut, use_container_width=True)
            else:
                st.info("No data available.")

        with c2:
            st.subheader("Airline Price Analysis")
            if not df_filtered.empty:
                # Average Price per Airline
                avg_price_airline = df_filtered.groupby('airline')['price'].mean().reset_index().sort_values(by='price', ascending=False)
                fig_bar_airline = px.bar(avg_price_airline, x='airline', y='price', color='price', 
                                 title="Average Price per Airline",
                                 labels={'price': 'Avg Price', 'airline': 'Airline'},
                                 color_continuous_scale='Blues')
                st.plotly_chart(fig_bar_airline, use_container_width=True)
            else:
                st.info("No data available.")

        # Row 2: Duration vs Price
        st.subheader("Duration vs Price")
        if not df_filtered.empty:
            fig_scatter = px.scatter(df_filtered, x='duration', y='price', color='airline', hover_data=['class'], title="Flight Duration vs Price")
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Row 3: Stops and Class (Side by Side)
        r3c1, r3c2 = st.columns(2)
        
        with r3c1:
            st.subheader("Price Analysis by Stops")
            if not df_filtered.empty:
                avg_price_stops = df_filtered.groupby('stops')['price'].mean().reset_index()
                fig_bar_stops = px.bar(avg_price_stops, x='stops', y='price', color='price', 
                                 title="Average Price per Stop Config",
                                 labels={'price': 'Avg Price', 'stops': 'Stops'},
                                 color_continuous_scale='Blues')
                st.plotly_chart(fig_bar_stops, use_container_width=True)
            else:
                st.info("No data available.")
        
        with r3c2:
            st.subheader("Price Distribution by Class")
            if not df_filtered.empty:
                 fig_box = px.box(df_filtered, x='class', y='price', color='class', points=False, title="Price Range by Class")
                 st.plotly_chart(fig_box, use_container_width=True)

    else:
        st.error("Data could not be loaded.")

elif page == "Predict":
    st.title("💸 Flight Price Prediction")
    st.write("Enter flight details below to get an estimated fare.")
    
    if df_raw is not None and pipeline is not None:
         with st.form("prediction_form"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                # Use raw df for full options list even if filters applied on other tab
                p_airline = st.selectbox("Airline", options=df_raw['airline'].unique())
                p_source = st.selectbox("Source City", options=df_raw['source_city'].unique())
                p_dest = st.selectbox("Destination City", options=df_raw['destination_city'].unique())
                p_stops = st.selectbox("Stops", options=df_raw['stops'].unique())
                
            with col_b:
                p_class = st.selectbox("Class", options=df_raw['class'].unique())
                p_duration = st.number_input("Duration (hours)", min_value=1.0, value=2.0, step=0.5)
                # Date and Time Inputs
                p_date = st.date_input("Travel Date", min_value=pd.Timestamp.today())
                p_time = st.time_input("Departure Time", value=pd.Timestamp.now().time())
            
            submitted = st.form_submit_button("Predict Price", type="primary")
            
            if submitted:
                # 1. Calculate Days Left
                today = pd.Timestamp.today().normalize()
                travel_date = pd.Timestamp(p_date).normalize()
                days_left = (travel_date - today).days + 1 # +1 to avoid 0 if same day
                
                # 2. Map Time to 'departure_time' category and 'Dep_Hour' value
                hour = p_time.hour
                
                # Logic to map exact hour to the categorical buckets used in training
                # 'Early_Morning': 5, 'Morning': 9, 'Afternoon': 14, 'Evening': 19, 'Night': 22, 'Late_Night': 2
                if 2 <= hour < 5:
                    dep_time_cat = "Late_Night"
                    dep_hour_val = 2
                elif 5 <= hour < 9:
                    dep_time_cat = "Early_Morning"
                    dep_hour_val = 5
                elif 9 <= hour < 14:
                    dep_time_cat = "Morning"
                    dep_hour_val = 9
                elif 14 <= hour < 19:
                    dep_time_cat = "Afternoon"
                    dep_hour_val = 14
                elif 19 <= hour < 22:
                    dep_time_cat = "Evening"
                    dep_hour_val = 19
                else: # 22 <= hour or hour < 2 (Night)
                    dep_time_cat = "Night"
                    dep_hour_val = 22

                # Combine inputs into DataFrame
                input_data = pd.DataFrame({
                    'airline': [p_airline],
                    'source_city': [p_source],
                    'destination_city': [p_dest],
                    'stops': [p_stops],
                    'class': [p_class],
                    'duration': [p_duration],
                    'days_left': [days_left],
                    'departure_time': [dep_time_cat],
                    'arrival_time': [dep_time_cat], # Dummy
                    'Dep_Hour': [dep_hour_val]
                })
                
                try:
                    pred_price = pipeline.predict(input_data)[0]
                    st.success(f"### Estimated Flight Price: ₹ {pred_price:,.2f}")
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.warning("Model or Data not available. Please ensure model training was successful.")

elif page == "About Me":
    st.title("👨‍💻 About Me")
    
    st.markdown("### Mouad Bakhchane")
    st.markdown("Data Scientist & Developer")
    
    st.markdown("---")
    st.markdown("**🌐 Website:** [www.mouadbakhchane.com](http://www.mouadbakhchane.com)")
    st.markdown("**🔗 LinkedIn:** [Connect on LinkedIn](https://www.linkedin.com/in/mouad-bakhchane)") # Placeholder link, user didn't provide specific URL structure
    st.markdown("**🐙 GitHub:** [View on GitHub](https://github.com/MouadBAKHCHANE)") # Placeholder link
    
    st.success("Feel free to connect effectively!")

# --- Sidebar Footer ---
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; margin-top: 20px;'>
            <p style='margin-bottom: 5px; color: gray; font-size: 0.9em;'>Indian Aviation Market Analysis</p>
            <p style='font-size: 1em;'>
                Made by <a href='http://www.mouadbakhchane.com' target='_blank' style='font-family: "Brush Script MT", cursive; font-size: 1.3em; text-decoration: none;'>Mouad</a>
            </p>
        </div>
    """, unsafe_allow_html=True)