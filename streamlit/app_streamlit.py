# streamlit/app_streamlit.py

import streamlit as st
import requests
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import sys
import os

if "df" not in st.session_state:
    st.session_state.df = None

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .fraud-alert {
        background-color: #ffcccc;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff3333;
    }
    .safe-transaction {
        background-color: #00BC66;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #33cc33;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown("<h1 class='main-header'>🕵️ Fraud Detection Dashboard</h1>", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2455/2455234.png", width=100)
    st.title("Configuration")
    
    # Api endpoint
    api_endpoint = st.text_input(
        "API Endpoint",
        value="http://localhost:8000",
        help="URL of the FastAPI backend"
    )
    
    # Prediction threshold
    threshold = st.slider(
        "Fraud Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Probability threshold for fraud classification"
    )
    
    # Update threshold button
    if st.button("Update Threshold"):
        try:
            response = requests.post(
                f"{api_endpoint}/threshold",
                json={"threshold": threshold}
            )
            if response.status_code == 200:
                st.success(f"Threshold updated to {threshold}")
            else:
                st.error("Failed to update threshold")
        except Exception as e:
            st.error(f"Error: {str(e)}")
            
    st.divider()
    
    # Model info
    if st.button("Get Model Info"):
        try:
            response = requests.get(f"{api_endpoint}/model/info")
            if response.status_code == 200:
                info = response.json()
                st.json(info)
            else:
                st.error(f"API Error: {response.status_code}")
        except Exception as e:
            st.error(f"Connection Error: {str(e)}")
            
    st.divider()
    st.markdown("### About")
    st.markdown("""
    This dashboard provides real-time fraud detection
    for financial transactions using machine learning.
    
    **Features:**
    - Single transaction prediction
    - Batch processing
    - Real-time visualization
    - Risk assessment
    """)
    
    # Health check
    try:
        health_response = requests.get(f"{api_endpoint}/health")
        if health_response.status_code == 200:
            st.success("✅ API is healthy")
        else:
            st.error("❌ API is not responding")
    except:
        st.error("❌ Cannot connect to API")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard", 
    "🔍 Single Prediction", 
    "📁 Batch Processing", 
    "📈 Analytics"
])

# Helper functions
def call_api(endpoint, method="GET", data=None):
    """Helper function to call API"""
    try:
        if method == "GET":
            response = requests.get(f"{api_endpoint}{endpoint}")
        elif method == "POST":
            response = requests.post(f"{api_endpoint}{endpoint}", json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None

# Tab 1: Dashboard
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Current Threshold", f"{threshold:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        model_info = call_api("/model/info")
        if model_info:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Model Type", model_info.get("model_type", "Unknown"))
            st.markdown("</div>", unsafe_allow_html=True)
        
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Features", "287")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Accuracy", "98.8%")
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.divider()
    
    # Recent predictions (mock data for now)
    st.subheader("Recent Activity")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'Transaction ID': ['TXN_' + str(i) for i in range(1001, 1011)],
        'Amount': np.random.randint(100, 100000, 10),
        'Probability': np.random.random(10),
        'Time': pd.date_range(start='2024-01-01', periods=10, freq='H')
    })
    sample_data['Fraud'] = sample_data['Probability'] > threshold
    sample_data['Risk'] = pd.cut(
        sample_data['Probability'],
        bins=[0, 0.3, 0.5, 0.8, 1.0],
        labels=['Low', 'Medium', 'High', 'Critical'],
        include_lowest=True
    )

    sample_data = sample_data.dropna(subset=["Risk"])
    sample_data["Risk"] = sample_data["Risk"].astype(str)
    
    # Display table
    st.dataframe(sample_data[['Transaction ID', 'Amount', 'Probability', 'Risk', 'Time']],
                use_container_width=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Fraud distribution
        fraud_counts = sample_data['Fraud'].value_counts()
        fig1 = px.pie(
            values=fraud_counts.values,
            names=['Non-Fraud', 'Fraud'],
            title='Fraud Distribution',
            color_discrete_sequence=['green', 'red']
        )
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        # Amount vs Probability
        sample_data["Risk"] = sample_data["Risk"].astype(str)
        fig2 = px.scatter(
            sample_data,
            x='Amount',
            y='Probability',
            color='Risk',
            hover_data=['Transaction ID'],
            title='Transaction Risk Analysis',
            color_discrete_map={
                'Low': 'green',
                'Medium': 'yellow',
                'High': 'orange',
                'Critical': 'red'
            }
        )
        fig2.update_traces(marker=dict(size=10))
        st.plotly_chart(fig2, use_container_width=True)

# Tab 2: Single Prediction
with tab2:
    st.subheader("Single Transaction Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.form("single_prediction_form"):
            st.markdown("### Transaction Details")
            amount = st.number_input("Amount", min_value=0.0, value=1500.0, step=100.0)
            customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=35)
            browser = st.selectbox(
                "Browser",
                ["Chrome_Some(66)", "Chrome_Some(77)", "Firefox", "Safari", "Edge", "Unknown"]
            )
            channel = st.selectbox(
                "Channel",
                ["channel_A", "channel_B", "channel_C", "mobile", "web"]
            )
            
            # Date inputs
            col_date1, col_date2, col_date3 = st.columns(3)
            with col_date1:
                date_input = st.date_input("Date")
            with col_date2:
                time_input = st.time_input("Time")
            with col_date3:
                txn_date = datetime.combine(date_input, time_input).isoformat() if date_input and time_input else datetime.now().isoformat()
            
            submitted = st.form_submit_button("Predict Fraud")
    
    with col2:
        st.markdown("### Sample Transaction")
        
        sample_json = {
            "uuid": "sample_001",
            "amount": 120000.0,
            "customer_age": 51.0,
            "browser": "Chrome_Some(66)",
            "channel": "channel_B",
            "ip": "102.253.144.146",
            "date": txn_date,
            "loginTime": txn_date,
            "txn_timestamp": txn_date
        }
        
        st.json(sample_json)
        
        if st.button("Load Sample"):
            st.session_state.sample_loaded = sample_json
    
    # Handle prediction
    if submitted or 'sample_loaded' in st.session_state:
        if 'sample_loaded' in st.session_state:
            transaction_data = st.session_state.sample_loaded
            del st.session_state.sample_loaded
        else:
            transaction_data = {
                "uuid": f"txn_{int(time.time())}",
                "amount": amount,
                "customer_age": float(customer_age),
                "browser": browser,
                "channel": channel,
                "date": txn_date,
                "loginTime": txn_date,
                "txn_timestamp": txn_date
            }
        
        # Show loading
        with st.spinner("Analyzing transaction for fraud..."):
            result = call_api("/predict", method="POST", data=transaction_data)
        
        if result:
            st.divider()
            st.markdown("### Prediction Results")
            
            # Display results
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                prob = result.get('fraud_probability', 0)
                is_fraud = result.get('is_fraud', False)
                st.markdown(f"""
                <div class='{"fraud-alert" if is_fraud else "safe-transaction"}'>
                    <h3>{"⚠️ FRAUD DETECTED" if is_fraud else "✅ SAFE TRANSACTION"}</h3>
                    <p>Probability: <b>{prob:.3f}</b></p>
                    <p>Threshold: <b>{result.get('threshold', 0.5)}</b></p>
                    <p>Risk Level: <b>{result.get('risk_level', 'Unknown')}</b></p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_res2:
                # Probability gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Fraud Probability %"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': threshold * 100
                        }
                    }
                ))
                fig_gauge.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col_res3:
                st.markdown("#### Transaction Details")
                st.json(transaction_data)

# Tab 3: Batch Processing
with tab3:
    st.subheader("Batch Transaction Processing")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV file with transactions",
        type=['csv', 'xlsx'],
        help="CSV should contain columns: amount, customer_age, browser, channel"
    )
    
    if uploaded_file is not None:
        # Load data
        if uploaded_file.name.endswith('.csv'):
            st.session_state.df = pd.read_csv(uploaded_file)
        else:
            st.session_state.df = pd.read_excel(uploaded_file)
        
        st.markdown(f"**Loaded {len(df)} transactions**")
        st.dataframe(df.head(), use_container_width=True)
        
        # Convert to API format
        transactions = []
        for idx, row in df.iterrows():
            now_iso = datetime.now().isoformat()

            transaction = {
                "uuid": f"upload_{idx}_{int(time.time())}",
                "amount": float(row.get('amount', 0)),
                "customer_age": float(row.get('customer_age', 30)),
                "browser": str(row.get('browser', 'Chrome_Some(66)')),
                "channel": str(row.get('channel', 'channel_A')),
                "date": now_iso,
                "loginTime": now_iso,
                "txn_timestamp": now_iso
            }
            transactions.append(transaction)
        
        # Batch prediction
        if st.button("Process Batch", type="primary"):
            batch_data = {
                "transactions": transactions,
                "threshold": threshold
            }
            
            with st.spinner(f"Processing {len(transactions)} transactions..."):
                result = call_api("/predict/batch", method="POST", data=batch_data)
            
            if result:
                st.success(f"✅ Batch processed successfully!")
                
                # Display statistics
                stats = result.get('statistics', {})
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                
                with col_stat1:
                    st.metric("Total", stats.get('total_transactions', 0))
                with col_stat2:
                    st.metric("Fraud Count", stats.get('fraud_count', 0))
                with col_stat3:
                    fraud_pct = stats.get('fraud_percentage', 0)
                    st.metric("Fraud %", f"{fraud_pct:.2f}%")
                with col_stat4:
                    avg_prob = stats.get('avg_fraud_probability', 0)
                    st.metric("Avg Probability", f"{avg_prob:.3f}")
                
                # Display results
                predictions_df = pd.DataFrame(result.get('predictions', []))
                if not predictions_df.empty:
                    st.dataframe(predictions_df, use_container_width=True)
                    
                    # Download results
                    csv = predictions_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"fraud_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
    
    else:
        # Manual batch input
        st.markdown("### Or enter transactions manually")
        
        num_transactions = st.slider("Number of transactions", 1, 10, 3)
        
        # Create a list to store transaction inputs
        transaction_inputs = []
        
        # Create form for each transaction
        for i in range(num_transactions):
            with st.expander(f"Transaction {i+1}", expanded=True if i < 3 else False):
                col_a, col_b = st.columns(2)
                with col_a:
                    amount_i = st.number_input(f"Amount", min_value=0.0, value=1000.0, key=f"amt_{i}")
                    age_i = st.number_input(f"Age", min_value=18, value=35, key=f"age_{i}")
                with col_b:
                    browser_i = st.selectbox(f"Browser", 
                                           ["Chrome_Some(66)", "Chrome_Some(77)", "Firefox", "Safari", "Edge", "Unknown"], 
                                           key=f"browser_{i}")
                    channel_i = st.selectbox(f"Channel", 
                                           ["channel_A", "channel_B", "channel_C", "mobile", "web"], 
                                           key=f"channel_{i}")
                
                # Store this transaction's data
                now_iso = datetime.now().isoformat()
                transaction_inputs.append({
                    "uuid": f"manual_{i}_{int(time.time())}",
                    "amount": float(amount_i),
                    "customer_age": float(age_i),
                    "browser": browser_i,
                    "channel": channel_i,
                    "date": now_iso,
                    "loginTime": now_iso,
                    "txn_timestamp": now_iso
                })

        # Predict Batch button
        if st.button("Predict Batch", type="primary"):
            # Format for API
            batch_data = {
                "transactions": transaction_inputs,
                "threshold": threshold
            }
            
            with st.spinner(f"Processing {num_transactions} transactions..."):
                result = call_api("/predict/batch", method="POST", data=batch_data)
            
            if result:
                st.success(f"✅ Batch processed successfully!")
                
                # Display statistics
                stats = result.get('statistics', {})
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                
                with col_stat1:
                    st.metric("Total", stats.get('total_transactions', 0))
                with col_stat2:
                    st.metric("Fraud Count", stats.get('fraud_count', 0))
                with col_stat3:
                    fraud_pct = stats.get('fraud_percentage', 0)
                    st.metric("Fraud %", f"{fraud_pct:.2f}%")
                with col_stat4:
                    avg_prob = stats.get('avg_fraud_probability', 0)
                    st.metric("Avg Probability", f"{avg_prob:.3f}")
                
                # Display results
                predictions_df = pd.DataFrame(result.get('predictions', []))
                if not predictions_df.empty:
                    st.dataframe(predictions_df, use_container_width=True)
                    
                    # Download results
                    csv = predictions_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"fraud_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            else:
                st.error("❌ Failed to process batch")

# Tab 4: Analytics
with tab4:
    st.subheader("Model Analytics & Insights")
    
    col_anal1, col_anal2 = st.columns(2)
    
    with col_anal1:
        st.markdown("### Model Performance")
        
        # Performance metrics
        metrics = {
            "Accuracy": 98.77,
            "Precision": 95.00,
            "Recall": 85.38,
            "F1-Score": 89.89,
            "AUC-ROC": 99.20
        }
        
        for metric, value in metrics.items():
            st.progress(value/100, text=f"{metric}: {value}%")
        
        # Feature importance visualization
        st.markdown("### Feature Importance")
        
        # Sample feature importance data
        feature_data = pd.DataFrame({
            'Feature': ['Transaction Amount', 'Account Volume', 'Customer Age', 
                       'Browser Type', 'Payment Type', 'Region', 'Time of Day'],
            'Importance': [25, 20, 15, 10, 8, 7, 5]
        })
        
        fig_importance = px.bar(
            feature_data,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Top Features Influencing Fraud Detection",
            color='Importance',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    
    with col_anal2:
        st.markdown("### Risk Distribution Analysis")
        
        # Sample risk distribution
        risk_data = pd.DataFrame({
            'Risk Level': ['Low (0-0.3)', 'Medium (0.3-0.5)', 'High (0.5-0.8)', 'Critical (0.8-1)'],
            'Count': [85, 10, 4, 1],
            'Color': ['green', 'yellow', 'orange', 'red']
        })
        
        fig_risk = px.pie(
            risk_data,
            values='Count',
            names='Risk Level',
            title="Transaction Risk Distribution",
            color='Risk Level',
            color_discrete_map={'Low (0-0.3)': 'green', 
                                'Medium (0.3-0.5)': 'yellow',
                                'High (0.5-0.8)': 'orange',
                                'Critical (0.8-1)': 'red'}
        )
        st.plotly_chart(fig_risk, use_container_width=True)
        
        # Time series analysis
        st.markdown("### Fraud Trends Over Time")
        
        # Sample time series data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        fraud_counts = np.random.randint(0, 20, 30)
        
        fig_trend = px.line(
            x=dates,
            y=fraud_counts,
            title="Daily Fraud Detection Count",
            labels={'x': 'Date', 'y': 'Fraud Count'}
        )
        fig_trend.update_traces(line_color='red', line_width=2)
        st.plotly_chart(fig_trend, use_container_width=True)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.9rem;">
    <p>Fraud Detection System v1.0 | Powered by Machine Learning</p>
    <p>For support contact: admin@fraud-detection.com</p>
</div>
""", unsafe_allow_html=True)