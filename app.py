import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from reportlab.pdfgen import canvas
import io
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="AI Analytics Visualization Platform",
    page_icon="📊",
    layout="wide"
)
st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

h1 {
    color: #1E88E5;
}

div[data-testid="metric-container"] {
    background-color: #f5f5f5;
    border: 1px solid #ddd;
    padding: 15px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

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
    ### Intelligent Data Analysis Made Simple

    Upload datasets, preprocess data, create visualizations,
    train machine learning models, generate AI insights,
    and download professional reports.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Dataset Analysis",
            "Complete"
        )

    with col2:
        st.metric("Visualizations", "6+")

    with col3:
        st.metric("ML Models", "2")

    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric(
            "AI Insights",
            "Available"
        )

    with col5:
        st.metric(
            "Reports",
            "PDF"
        )

    with col6:
        st.metric(
            "Outlier Detection",
            "IQR"
        )
    st.divider()

    st.subheader("🚀 Platform Features")

    st.markdown("""
    - 📂 Dataset Upload & Analysis
    - 🧹 Data Preprocessing
    - 📊 Interactive Visualizations
    - 🤖 Machine Learning Models
    - 💡 AI-Based Insights
    - 📄 PDF Report Generation
    """)

    st.markdown("---")

    st.caption(
        "Developed by Wahida Begum | BCA Final Year Project"
    )
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

            st.session_state["df"] = df

            st.subheader("👀 Dataset Preview")
            st.dataframe(df.head(10))
            st.divider()
        

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Rows", df.shape[0])

            with col2:
                st.metric("Columns", df.shape[1])

            with col3:
                st.metric(
                    "Missing Values",
                    int(df.isnull().sum().sum())
                )

            with col4:
                st.metric(
                    "Duplicates",
                    int(df.duplicated().sum())
                )

            st.subheader("📝 Column Names")
            st.write(list(df.columns))

            st.subheader("🔍 Data Types")
            st.dataframe(df.dtypes.astype(str))

            st.subheader("📈 Basic Statistics")
            st.dataframe(df.describe())

            st.subheader("🔍 Missing Values")

            missing = df.isnull().sum()

            st.dataframe(
                missing[missing > 0].reset_index().rename(
                    columns={"index": "Column", 0: "Missing Values"}
                )
            )


            st.subheader("💾 Dataset Memory Usage")

            memory_usage = df.memory_usage(deep=True).sum() / 1024

            st.metric(
                "Memory Usage",
                f"{memory_usage:.2f} KB"
            )

            numeric_cols = df.select_dtypes(include=["number"]).columns
            categorical_cols = df.select_dtypes(exclude=["number"]).columns

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🔢 Numeric Columns")
                st.write(list(numeric_cols))

            with col2:
                st.subheader("🔤 Categorical Columns")
                st.write(list(categorical_cols))

        except Exception as e:
            st.error(f"Error loading file: {e}")


# Data Preprocessing Page
elif menu == "🧹 Data Preprocessing":

    st.title("🧹 Data Preprocessing")

    if "df" not in st.session_state:
        st.warning("Please upload a dataset first.")
    else:

        df = st.session_state["df"]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Rows", df.shape[0])

        with col2:
            st.metric("Columns", df.shape[1])

        with col3:
            st.metric(
                "Missing Values",
                int(df.isnull().sum().sum())
            )

        with col4:
            st.metric(
                "Duplicate Rows",
                int(df.duplicated().sum())
            )

        st.subheader("🔍 Missing Values")

        missing = df.isnull().sum()

        missing_df = missing[missing > 0]

        if len(missing_df) > 0:

            st.dataframe(
                missing_df.reset_index().rename(
                    columns={"index": "Column", 0: "Missing Values"}
                )
            )

        else:

            st.success("✅ No missing values found.")

        st.subheader("🗑 Remove Duplicates")

        if st.button("Remove Duplicate Rows"):

            df = df.drop_duplicates()

            st.session_state["df"] = df

            st.success("Duplicate rows removed successfully!")

            st.write(f"New Shape: {df.shape}")

        st.subheader("🩹 Fill Missing Values")


        if st.button("Fill Missing Values"):

            for col in df.columns:

                if pd.api.types.is_numeric_dtype(df[col]):

                    df[col] = df[col].fillna(df[col].median())

                else:

                    mode_value = df[col].mode()

                    if not mode_value.empty:
                        df[col] = df[col].fillna(mode_value[0])

            st.session_state["df"] = df

            st.success("Missing values filled successfully!") 

        st.subheader("👀 Processed Dataset Preview")

        st.dataframe(df.head(10))

        st.subheader("📥 Download Processed Dataset")

        csv = df.to_csv(index=False)

        st.download_button(
            label="Download Cleaned Dataset",
            data=csv,
            file_name="processed_dataset.csv",
            mime="text/csv"
        )
        st.divider()

        st.subheader("📊 Outlier Detection")

        numeric_cols = df.select_dtypes(include="number").columns

        selected_col = st.selectbox(
            "Select Column for Outlier Detection",
            numeric_cols
        )

        Q1 = df[selected_col].quantile(0.25)

        Q3 = df[selected_col].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - (1.5 * IQR)

        upper_bound = Q3 + (1.5 * IQR)

        outliers = df[
            (df[selected_col] < lower_bound)
    |
            (df[selected_col] > upper_bound)
        ]

        st.metric(
            "Outliers Detected",
            len(outliers)
        )

        if len(outliers) > 0:

            st.dataframe(outliers)

        st.info(
            f"{len(outliers)} outliers detected in {selected_col}"
        )

# Visualization Page
elif menu == "📈 Visualization":

    st.title("📈 Data Visualization")

    if "df" not in st.session_state:

        st.warning("Please upload a dataset first.")

    else:

        df = st.session_state["df"]

        st.markdown(
            "Explore your dataset through interactive visualizations."
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Chart Types",
                "6 Available"
            )

        with col2:
            st.metric(
                "Numeric Columns",
                len(
                    df.select_dtypes(include="number").columns
                )
            )

        with col3:
            st.metric(
                "Categorical Columns",
                len(
                    df.select_dtypes(exclude="number").columns
                )
            )

        st.divider()

        st.subheader("📊 Select Visualization")

        st.caption(
            "Choose a chart type to explore patterns, distributions, and relationships in the dataset."
        )

        chart_type = st.selectbox(
            "Choose Chart Type",
            [
                "Histogram",
                "Bar Chart",
                "Pie Chart",
                "Box Plot",
                "Correlation Heatmap",
                "Scatter Plot"
            ]
        )
        st.divider()

        if chart_type == "Histogram":

            numeric_cols = df.select_dtypes(include="number").columns

            column = st.selectbox(
                "Select Numeric Column",
                numeric_cols
            )

            st.info(
                f"Showing distribution of {column}"
            )
            fig = px.histogram(
                df,
                x=column,
                title=f"Distribution of {column}"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Bar Chart":

            categorical_cols = df.select_dtypes(exclude="number").columns

            if len(categorical_cols) > 0:

                column = st.selectbox(
                    "Select Categorical Column",
                    categorical_cols
                )

                st.info(
                    f"Showing category distribution for {column}"
                )

                value_counts = df[column].value_counts().reset_index()

                value_counts.columns = [column, "Count"]

                fig = px.bar(
                    value_counts,
                    x=column,
                    y="Count",
                    title=f"{column} Distribution"
                )

                st.plotly_chart(fig, use_container_width=True)

            else:

                st.warning("No categorical columns found.")

        elif chart_type == "Pie Chart":

            categorical_cols = df.select_dtypes(exclude="number").columns

            if len(categorical_cols) > 0:

                column = st.selectbox(
                    "Select Categorical Column",
                    categorical_cols
                )

                st.info(
                    f"Showing proportion of each category in {column}"
                )
                fig = px.pie(
                    df,
                    names=column,
                    title=f"{column} Distribution"
                )

                st.plotly_chart(fig, use_container_width=True)

            else:

                st.warning("No categorical columns found.")   


        elif chart_type == "Box Plot":

            numeric_cols = df.select_dtypes(include="number").columns

            column = st.selectbox(
                "Select Numeric Column",
                numeric_cols
            )

            st.info(
                f"Detecting outliers in {column}"
            )
            fig = px.box(
                df,
                y=column,
                title=f"Box Plot of {column}"
            )

            st.plotly_chart(fig, use_container_width=True)             

        elif chart_type == "Correlation Heatmap":

            numeric_df = df.select_dtypes(include="number")

            corr_matrix = numeric_df.corr()

            st.info(
                "Correlation values close to +1 or -1 indicate strong relationships."
            )
            fig = go.Figure(
                data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns
                )
            )

            fig.update_layout(
                title="Correlation Heatmap"
            )

            st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Scatter Plot":

            numeric_cols = df.select_dtypes(include="number").columns

            x_col = st.selectbox(
                "Select X-Axis",
                numeric_cols,
                key="scatter_x"
            )

            y_col = st.selectbox(
                "Select Y-Axis",
                numeric_cols,
                key="scatter_y"
            )

            st.info(
                f"Analyzing relationship between {x_col} and {y_col}"
            )
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                title=f"{x_col} vs {y_col}"
            )

            st.plotly_chart(fig, use_container_width=True)

# Machine Learning Page
elif menu == "🤖 Machine Learning":

    st.title("🤖 Machine Learning")

    st.markdown(
        """
        Train machine learning models and evaluate their performance
        directly on your uploaded dataset.
        """
    )

    if "df" not in st.session_state:

        st.warning("Please upload a dataset first.")

    else:

        df = st.session_state["df"]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Rows",
                df.shape[0]
            )

        with col2:
            st.metric(
                "Columns",
                df.shape[1]
            )

        with col3:
            st.metric(
                "Models Available",
                "2"
            )

        st.divider()

        st.subheader("⚙️ Model Configuration")

        model_type = st.selectbox(
            "Select Model",
            [
                "Logistic Regression",
                "Linear Regression"
            ]
        )

        if model_type == "Logistic Regression":

            st.info(
                "Used for classification problems where the target contains categories."
            )

        else:

            st.info(
                "Used for predicting continuous numerical values."
            )

        if model_type == "Logistic Regression":

            possible_targets = [
                col for col in df.columns
                if df[col].nunique() <= 20
            ]

        else:

            possible_targets = list(df.columns)

        target_column = st.selectbox(
            "Select Target Column",
            possible_targets
        )

        if st.button("🚀 Train Model"):
            try:
                X = df.drop(columns=[target_column])

                y = df[target_column]

                if y.dtype == "object":

                    y = pd.factorize(y)[0]

                X = pd.get_dummies(
                    X,
                    drop_first=True
                )

                if X.isnull().sum().sum() > 0:

                    st.error(
                        "Dataset still contains missing values. Please use the Data Preprocessing module first."
                    )

                    st.stop()

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42
                )

                if model_type == "Logistic Regression":

                    model = LogisticRegression(max_iter=1000)

                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)

                    st.success(
                        "Model trained successfully."
                    )

                    accuracy = accuracy_score(
                        y_test,
                        y_pred
                    )

                    st.metric(
                        "Model Accuracy",
                        f"{accuracy * 100:.2f}%"
                    )

                    st.subheader("Confusion Matrix")

                    cm = confusion_matrix(
                        y_test,
                        y_pred
                    )

                    st.dataframe(cm)

                    st.subheader(
                        "Classification Report"
                    )

                    report = classification_report(
                        y_test,
                        y_pred,
                        output_dict=True
                    )

                    st.dataframe(
                        pd.DataFrame(report).transpose()
                    )
                
                else:

                    model = LinearRegression()

                    model.fit(
                        X_train,
                        y_train
                    )

                    y_pred = model.predict(
                        X_test
                    )

                    st.success(
                        "Model trained successfully."
                    )

                    score = model.score(
                        X_test,
                        y_test
                    )

                    st.metric(
                        "R² Score",
                        f"{score:.4f}"
                    )

            except Exception as e:

                st.error(
                    f"Error training model: {e}"
                )


# AI Insights Page
elif menu == "💡 AI Insights":

    st.title("💡 AI Insights")

    st.markdown(
        """
        Generate intelligent observations and statistical
        insights from your dataset automatically.
        """
    )

    if "df" not in st.session_state:
        st.warning("Please upload a dataset first.")

    else:

        df = st.session_state["df"]

        st.subheader("📋 Dataset Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Rows", df.shape[0])

        with col2:
            st.metric("Columns", df.shape[1])

        with col3:
            st.metric(
                "Missing Values",
                int(df.isnull().sum().sum())
            )

        with col4:
            st.metric(
                "Duplicates",
                int(df.duplicated().sum())
            )

        st.divider()

        st.subheader("📈 Numerical Insights")

        numeric_cols = df.select_dtypes(include="number").columns

        insights = []

        for col in numeric_cols:

            insights.append({
                "Column": col,
                "Mean": round(df[col].mean(), 2),
                "Min": df[col].min(),
                "Max": df[col].max()
            })

        st.dataframe(pd.DataFrame(insights))

        st.subheader("🤖 Smart Insights")

        if "BMI" in df.columns:

            avg_bmi = df["BMI"].mean()

            if avg_bmi > 25:
                st.warning(
                    f"Average BMI is {avg_bmi:.2f}. Population appears overweight."
                )

        if "BloodPressure" in df.columns:

            avg_bp = df["BloodPressure"].mean()

            if avg_bp > 130:
                st.warning(
                    f"Average Blood Pressure is {avg_bp:.2f}. Elevated BP detected."
                )

        if "Glucose" in df.columns:

            avg_glucose = df["Glucose"].mean()

            if avg_glucose > 125:
                st.warning(
                    f"Average Glucose is {avg_glucose:.2f}. High glucose levels observed."
                )

        if "Effected" in df.columns:

            affected_rate = df["Effected"].mean() * 100

            st.info(
                f"{affected_rate:.2f}% of records are marked as affected."
            )

        st.subheader("🔗 Correlation Insights")

        st.info(
            "Correlation values close to +1 or -1 indicate strong relationships between variables."
        )

        corr_matrix = df.select_dtypes(include="number").corr()

        st.dataframe(corr_matrix)

# Report Generation Page
elif menu == "📄 Report Generation":

    st.title("📄 Report Generation")

    st.markdown("""
    Generate a professional PDF report containing:

    - Dataset Summary
    - Missing Value Analysis
    - Duplicate Record Information
    - Numerical Insights
    - Statistical Overview
    """)

    if "df" not in st.session_state:

        st.warning("Please upload a dataset first.")

    else:

        df = st.session_state["df"]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Rows", df.shape[0])

        with col2:
            st.metric("Columns", df.shape[1])

        with col3:
            st.metric(
                "Missing Values",
                int(df.isnull().sum().sum())
            )

        with col4:
            st.metric(
                "Duplicates",
                int(df.duplicated().sum())
            )

        st.divider()

        st.info(
            "Click the button below to generate a downloadable PDF analytics report."
        )
        
        if st.button("📄 Generate PDF Report"):

            buffer = io.BytesIO()

            pdf = canvas.Canvas(buffer)

            y = 800

            pdf.setFont("Helvetica-Bold", 16)
            pdf.drawString(
                50,
                y,
                "AI Analytics Visualization Report"
            )

            y -= 40

            pdf.setFont("Helvetica", 12)

            pdf.drawString(
                50,
                y,
                f"Generated On: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
            )

            y -= 40

            pdf.setFont("Helvetica-Bold", 14)

            pdf.drawString(
                50,
                y,
                "Dataset Summary"
            )

            y -= 25

            pdf.setFont("Helvetica", 12)

            pdf.drawString(
                50,
                y,
                f"Rows: {df.shape[0]}"
            )

            y -= 20

            pdf.drawString(
                50,
                y,
                f"Columns: {df.shape[1]}"
            )

            y -= 20

            pdf.drawString(
                50,
                y,
                f"Missing Values: {df.isnull().sum().sum()}"
            )

            y -= 20

            pdf.drawString(
                50,
                y,
                f"Duplicate Rows: {df.duplicated().sum()}"
            )

            y -= 40

            pdf.setFont("Helvetica-Bold", 14)

            pdf.drawString(
                50,
                y,
                "Numerical Insights"
            )

            y -= 25

            pdf.setFont("Helvetica", 12)

            numeric_cols = df.select_dtypes(
                include="number"
            ).columns

            for col in numeric_cols:

                pdf.drawString(
                    50,
                    y,
                    f"{col}: Mean={df[col].mean():.2f}, Min={df[col].min()}, Max={df[col].max()}"
                )

                y -= 20

                if y < 50:
                    pdf.showPage()
                    y = 800

            pdf.save()

            buffer.seek(0)

            st.success("PDF report generated successfully.")

            st.download_button(
                label="📥 Download PDF Report",
                data=buffer,
                file_name="analytics_report.pdf",
                mime="application/pdf"
            )


# About Page
elif menu == "ℹ️ About":

    st.title("ℹ️ About")

    st.markdown("""
    ### AI Analytics Visualization Platform

    An AI-powered analytics application developed as a
    BCA Final Year Project using Streamlit, Machine Learning,
    Data Visualization, and Business Intelligence techniques.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Modules",
            "7"
        )

    with col2:
        st.metric(
            "Visualizations",
            "6"
        )

    with col3:
        st.metric(
            "ML Models",
            "2"
        )

    st.divider()

    st.subheader("🛠 Technologies Used")

    st.markdown("""
    - Python
    - Streamlit
    - Pandas
    - Plotly
    - Scikit-Learn
    - ReportLab
    - Git & GitHub
    """)

    st.subheader("🚀 Features")

    st.markdown("""
    - 📂 Dataset Analysis
    - 🧹 Data Preprocessing
    - 📈 Interactive Visualizations
    - 🤖 Machine Learning Models
    - 💡 AI Insights
    - 📄 PDF Report Generation
    """)

    st.divider()

    st.subheader("👩‍💻 Developed By")

    st.success(
        "Wahida Begum | Bachelor of Computer Applications (BCA) | Final Year Project"
    )