import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV file
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\user\Desktop\KIFIYA Projects\TellCo-Week-02\notebooks\merged_data.csv")

# Read the data
df = load_data()

# Get top 5 users by session_frequency
top_users = df.nlargest(5, 'session_frequency')

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["User Overview Analysis", "User Engagement Analysis", "Experience Analysis", "Satisfaction Analysis"])

# User Overview Analysis
if page == "User Overview Analysis":
    st.title("User Overview Analysis")
    st.dataframe(top_users)

    # Plotting session frequency
    plt.figure(figsize=(10, 5))
    plt.bar(top_users['IMSI'].astype(str), top_users['session_frequency'], color='blue')
    plt.title("Session Frequency of Top 5 Users")
    plt.xlabel("User IMSI")
    plt.ylabel("Session Frequency")
    st.pyplot(plt)

# User Engagement Analysis
elif page == "User Engagement Analysis":
    st.title("User Engagement Analysis")
    
    # Plotting engagement score
    plt.figure(figsize=(10, 5))
    plt.bar(top_users['IMSI'].astype(str), top_users['engagement_score'], color='green')
    plt.title("Engagement Score of Top 5 Users")
    plt.xlabel("User IMSI")
    plt.ylabel("Engagement Score")
    st.pyplot(plt)

# Experience Analysis
elif page == "Experience Analysis":
    st.title("Experience Analysis")
    
    # Plotting experience score
    plt.figure(figsize=(10, 5))
    plt.bar(top_users['IMSI'].astype(str), top_users['experience_score'], color='orange')
    plt.title("Experience Score of Top 5 Users")
    plt.xlabel("User IMSI")
    plt.ylabel("Experience Score")
    st.pyplot(plt)

# Satisfaction Analysis
elif page == "Satisfaction Analysis":
    st.title("Satisfaction Analysis")

    # Plotting satisfaction score
    plt.figure(figsize=(10, 5))
    plt.bar(top_users['IMSI'].astype(str), top_users['satisfaction_score'], color='red')
    plt.title("Satisfaction Score of Top 5 Users")
    plt.xlabel("User IMSI")
    plt.ylabel("Satisfaction Score")
    st.pyplot(plt)