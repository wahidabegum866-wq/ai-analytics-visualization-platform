import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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

            st.session_state["df"] = df

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

            st.subheader("🔍 Missing Values")

            missing = df.isnull().sum()

            st.dataframe(
                missing[missing > 0].reset_index().rename(
                    columns={"index": "Column", 0: "Missing Values"}
                )
            )

            st.subheader("📋 Duplicate Records")

            duplicates = df.duplicated().sum()

            st.write(f"Duplicate Rows: {duplicates}")


            st.subheader("💾 Dataset Memory Usage")

            memory_usage = df.memory_usage(deep=True).sum() / 1024

            st.write(f"{memory_usage:.2f} KB")

            st.subheader("📂 Column Classification")

            numeric_cols = df.select_dtypes(include=["number"]).columns
            categorical_cols = df.select_dtypes(exclude=["number"]).columns

            st.write("Numeric Columns:")
            st.write(list(numeric_cols))

            st.write("Categorical Columns:")
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

        st.subheader("Dataset Shape")
        st.write(df.shape)

        st.subheader("Missing Values")

        missing = df.isnull().sum()
        st.dataframe(
            missing[missing > 0].reset_index().rename(
                columns={"index": "Column", 0: "Missing Values"}
            )
        )

        st.subheader("Duplicate Rows")

        duplicates = df.duplicated().sum()
        st.write(f"Duplicate Rows: {duplicates}")

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


# Visualization Page
elif menu == "📈 Visualization":

    st.title("📈 Data Visualization")

    if "df" not in st.session_state:

        st.warning("Please upload a dataset first.")

    else:

        df = st.session_state["df"]

        chart_type = st.selectbox(
            "Select Chart Type",
            [
                "Histogram",
                "Bar Chart",
                "Pie Chart",
                "Box Plot",
                "Correlation Heatmap",
                "Scatter Plot"
            ]
        )

        if chart_type == "Histogram":

            numeric_cols = df.select_dtypes(include="number").columns

            column = st.selectbox(
                "Select Numeric Column",
                numeric_cols
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

            fig = px.box(
                df,
                y=column,
                title=f"Box Plot of {column}"
            )

            st.plotly_chart(fig, use_container_width=True)             

        elif chart_type == "Correlation Heatmap":

            numeric_df = df.select_dtypes(include="number")

            corr_matrix = numeric_df.corr()

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