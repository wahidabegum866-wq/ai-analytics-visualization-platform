import streamlit as st
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="AI Analytics Visualization Platform",
    page_icon="📊",
    layout="wide"
)

# Sidebar
st.sidebar.title("📊 AI Analytics Platform")

menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📂 Dataset",
        "🧹 Data Preprocessing",
        "📈 Visualization",
        "🤖 Machine Learning",
        "💡 AI Insights",
        "📄 Report Generation",
        "ℹ️ About"
    ]
)

# Home Page
if menu == "🏠 Home":
    st.title("📊 AI Analytics Visualization Platform")

    st.markdown("""
    Welcome to the AI Analytics Visualization Platform.

    This application will help users:

    - Upload datasets
    - Analyze data
    - Visualize insights
    - Build machine learning models
    - Generate reports
    - Obtain AI-powered recommendations

    Select a module from the sidebar to begin.
    """)

# Dataset Page
elif menu == "📂 Dataset":

    st.title("📂 Dataset Upload Module")

    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel file",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:

        try:

            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)

            else:
                df = pd.read_excel(uploaded_file)

            st.success("Dataset uploaded successfully!")

            st.subheader("👀 Dataset Preview")
            st.dataframe(df.head(10))

            st.subheader("📊 Dataset Shape")
            st.write(f"Rows: {df.shape[0]}")
            st.write(f"Columns: {df.shape[1]}")

            st.subheader("📝 Column Names")
            st.write(list(df.columns))

            st.subheader("🔍 Data Types")
            st.dataframe(df.dtypes.astype(str))

            st.subheader("📈 Basic Statistics")
            st.dataframe(df.describe())

        except Exception as e:
            st.error(f"Error loading file: {e}")

# Data Preprocessing Page
elif menu == "🧹 Data Preprocessing":
    st.title("🧹 Data Preprocessing")
    st.info("Coming Soon...")

# Visualization Page
elif menu == "📈 Visualization":
    st.title("📈 Visualization")
    st.info("Coming Soon...")

# Machine Learning Page
elif menu == "🤖 Machine Learning":
    st.title("🤖 Machine Learning")
    st.info("Coming Soon...")

# AI Insights Page
elif menu == "💡 AI Insights":
    st.title("💡 AI Insights")
    st.info("Coming Soon...")

# Report Generation Page
elif menu == "📄 Report Generation":
    st.title("📄 Report Generation")
    st.info("Coming Soon...")

# About Page
elif menu == "ℹ️ About":
    st.title("ℹ️ About")
    st.write("""
    Developed as a BCA Final Year AI Project.

    Technologies Used:
    - Python
    - Streamlit
    - Pandas
    - Plotly
    - Scikit-Learn
    """)