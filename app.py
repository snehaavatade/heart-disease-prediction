
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import database

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="HeartCare AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM DARK THEME
# =====================================================

st.markdown("""
<style>

/* Main application */

.stApp {
    background: linear-gradient(135deg, #081321 0%, #101F32 100%);
    color: #EAF2F8;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background: #101E30;
    border-right: 1px solid #263B52;
}

[data-testid="stSidebar"] h1 {
    color: #45D9F5 !important;
    font-size: 27px;
}

[data-testid="stSidebar"] p {
    color: #A9BBD0;
}

/* Headings */

h1 {
    color: #45D9F5 !important;
    font-weight: 750 !important;
}

h2, h3 {
    color: #DDF7FF !important;
    font-weight: 650 !important;
}

p, li, label {
    color: #D5E1ED;
}

/* Hero section */

.hero {
    background: linear-gradient(120deg, #102C45, #123C54);
    border: 1px solid #22617B;
    border-radius: 20px;
    padding: 36px;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 38px;
    margin-bottom: 10px;
}

.hero p {
    color: #C1D8E8;
    font-size: 16px;
    line-height: 1.8;
}

/* Cards */

.custom-card {
    background: #15283D;
    border: 1px solid #29445E;
    border-radius: 15px;
    padding: 23px;
    min-height: 145px;
    margin-bottom: 15px;
}

.custom-card h3 {
    color: #48D9F2 !important;
    margin-bottom: 10px;
    font-size: 19px;
}

.custom-card p {
    color: #B9CADB;
    font-size: 14px;
    line-height: 1.7;
}

/* Metrics */

div[data-testid="stMetric"] {
    background: #15283D;
    border: 1px solid #29445E;
    border-radius: 14px;
    padding: 19px;
}

div[data-testid="stMetricLabel"] {
    color: #AFC4D8;
}

div[data-testid="stMetricValue"] {
    color: #48D9F2;
}

/* Buttons */

.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(90deg, #0891B2, #13B8D4);
    color: white;
    border: none;
    border-radius: 9px;
    font-weight: 650;
    padding: 10px 18px;
    transition: 0.2s;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #087E9B;
    color: white;
    border: none;
}

/* Inputs */

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #172B40;
    border-color: #34536D;
    color: white;
}

input, textarea {
    color: white !important;
}

/* Form */

[data-testid="stForm"] {
    background: #13263A;
    border: 1px solid #29445E;
    border-radius: 16px;
    padding: 22px;
}

/* Expanders */

[data-testid="stExpander"] {
    background: #15283D;
    border: 1px solid #29445E;
    border-radius: 12px;
}

/* Tables */

[data-testid="stDataFrame"] {
    border: 1px solid #29445E;
    border-radius: 12px;
}

/* Alerts */

[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Dividers */

hr {
    border-color: #29445E;
}

/* Footer */

.footer {
    text-align: center;
    color: #8298AE;
    padding: 20px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD PROJECT FILES
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load("best_heart_disease_model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")


@st.cache_data
def load_comparison():
    return pd.read_csv("model_comparison.csv")


model = load_model()
data = load_data()
results = load_comparison()

database.create_table()


# =====================================================
# REUSABLE SECTION HEADER
# =====================================================

def section_header(title, description=None):
    st.markdown(f"## {title}")

    if description:
        st.markdown(
            f"<p style='color:#AFC4D8'>{description}</p>",
            unsafe_allow_html=True
        )


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.markdown(
    """
    <div style="text-align:center;padding:15px 5px;">
        <div style="font-size:43px;">❤️</div>
        <h1 style="color:#45D9F5;">HeartCare AI</h1>
        <p style="font-size:13px;">
            Heart Health & Machine Learning
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "MAIN MENU",
    [
        "Home",
        "Heart Health Learning Centre",
        "Heart Disease Prediction",
        "Model Performance",
        "Dataset Analysis",
        "Prediction History"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <div class="custom-card">
        <h3>About HeartCare AI</h3>
        <p>
        An academic machine learning project designed
        to explain heart health concepts and explore
        heart disease dataset classification.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.caption(
    "Educational use only | Not a medical diagnostic system"
)


# =====================================================
# HOME PAGE
# =====================================================

if page == "Home":

    st.markdown("""
    <div class="hero">
        <h1>Welcome to HeartCare AI ❤️</h1>
        <p>
        Understanding heart health through education
        and machine learning.<br>
        Explore medical concepts, learn about important
        health parameters and discover how machine
        learning analyzes healthcare datasets.
        </p>
    </div>
    """, unsafe_allow_html=True)

    section_header(
        "Project Overview",
        "An educational platform combining heart health information "
        "with machine learning."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dataset Records", len(data))

    with col2:
        st.metric("Input Features", 13)

    with col3:
        st.metric("Machine Learning Models", 3)

    st.markdown("---")

    section_header("Explore HeartCare AI")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="custom-card">
            <h3>📚 Heart Health Learning</h3>
            <p>
            Understand the human heart, common heart conditions,
            important medical terms, risk factors and prevention.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="custom-card">
            <h3>📊 Dataset Analysis</h3>
            <p>
            Explore the Cleveland dataset, its features,
            records and target distribution.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="custom-card">
            <h3>🗂 Prediction History</h3>
            <p>
            View previously saved model outputs and
            download prediction records.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="custom-card">
            <h3>🤖 Machine Learning Prediction</h3>
            <p>
            Enter sample feature values and explore
            the output generated by the selected model.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="custom-card">
            <h3>📈 Model Performance</h3>
            <p>
            Compare Logistic Regression, Decision Tree
            and Random Forest using evaluation metrics.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="custom-card">
            <h3>🔬 Data-Driven Learning</h3>
            <p>
            Understand how data preprocessing, training
            and model evaluation work together.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    section_header("How the Project Works")

    workflow = st.columns(4)

    steps = [
        ("01", "Learn", "Understand heart health concepts."),
        ("02", "Explore", "Understand the dataset features."),
        ("03", "Predict", "Explore machine learning outputs."),
        ("04", "Analyze", "Study model performance.")
    ]

    for column, step in zip(workflow, steps):
        with column:
            st.markdown(f"""
            <div class="custom-card" style="text-align:center;">
                <h1>{step[0]}</h1>
                <h3>{step[1]}</h3>
                <p>{step[2]}</p>
            </div>
            """, unsafe_allow_html=True)


# =====================================================
# HEART HEALTH LEARNING CENTRE
# =====================================================

elif page == "Heart Health Learning Centre":

    st.markdown("""
    <div class="hero">
        <h1>📚 Heart Health Learning Centre</h1>
        <p>
        Learn about the human heart, heart conditions,
        medical terminology and the parameters used
        in our machine learning project.
        </p>
    </div>
    """, unsafe_allow_html=True)

    topics = [
        "1. Understanding the Human Heart",
        "2. What Is Heart Disease?",
        "3. How Does the Heart Work?",
        "4. Important Medical Terms",
        "5. Heart Disease Risk Factors",
        "6. Prevention and Heart Care",
        "7. Understanding Our 13 Prediction Features"
    ]

    topic = st.selectbox(
        "SELECT A LEARNING TOPIC",
        topics
    )

    st.markdown("---")

    if topic == topics[0]:

        section_header("Understanding the Human Heart")

        st.write("""
        The heart is a muscular organ that pumps blood
        throughout the human body.

        It supplies oxygen and nutrients to different organs
        and helps remove carbon dioxide and other waste products.
        """)

        section_header("The Four Chambers of the Heart")

        chambers = [
            ("Right Atrium", "Receives oxygen-poor blood returning from the body."),
            ("Right Ventricle", "Pumps oxygen-poor blood to the lungs."),
            ("Left Atrium", "Receives oxygen-rich blood from the lungs."),
            ("Left Ventricle", "Pumps oxygen-rich blood to the rest of the body.")
        ]

        for name, description in chambers:
            with st.expander(name):
                st.write(description)

        st.info(
            "The left ventricle has a thicker muscular wall "
            "because it pumps blood throughout the body."
        )

        section_header("Main Functions")

        st.markdown("""
        - Circulates blood throughout the body.
        - Delivers oxygen and nutrients.
        - Helps remove waste products.
        - Maintains blood circulation and pressure.
        """)

    elif topic == topics[1]:

        section_header("What Is Heart Disease?")

        st.write("""
        Heart disease refers to a group of conditions
        that affect the structure or functioning of the heart.
        """)

        conditions = {
            "Coronary Artery Disease": (
                "Occurs when the arteries supplying blood to "
                "the heart become narrowed or blocked, often "
                "due to plaque buildup."
            ),
            "Heart Attack": (
                "Occurs when blood flow to part of the heart "
                "muscle is severely reduced or blocked."
            ),
            "Heart Failure": (
                "A condition in which the heart cannot pump "
                "blood as effectively as the body needs."
            ),
            "Arrhythmia": (
                "An irregular heartbeat that may be too fast, "
                "too slow or irregular."
            )
        }

        for name, description in conditions.items():
            with st.expander(name):
                st.write(description)

        section_header("Possible Warning Signs")

        st.markdown("""
        - Chest discomfort or pain.
        - Shortness of breath.
        - Unusual tiredness.
        - Dizziness or fainting.
        - Discomfort in the arms, back, neck or jaw.
        """)

        st.error(
            "Possible heart attack symptoms require immediate "
            "medical attention. Do not use this application "
            "to assess an emergency."
        )

    elif topic == topics[2]:

        section_header("How Does the Heart Work?")

        st.write("""
        The heart works through a continuous cycle of
        contraction and relaxation.
        """)

        circulation = [
            ("Step 1: Blood enters the right side",
             "Oxygen-poor blood returns from the body and enters the right atrium."),
            ("Step 2: Blood travels to the lungs",
             "The right ventricle pumps blood to the lungs, where it receives oxygen."),
            ("Step 3: Blood enters the left side",
             "Oxygen-rich blood returns from the lungs and enters the left atrium."),
            ("Step 4: Blood travels throughout the body",
             "The left ventricle pumps oxygen-rich blood to the rest of the body.")
        ]

        for title, description in circulation:
            st.markdown(f"""
            <div class="custom-card">
                <h3>{title}</h3>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)

        section_header("Understanding Heartbeat")

        st.write("""
        A heartbeat is produced by the coordinated contraction
        and relaxation of the heart muscles.

        Heart rate is generally measured in beats per minute (bpm).
        """)

    elif topic == topics[3]:

        section_header("Important Medical Terms")

        terms = {
            "Blood Pressure": (
                "The force exerted by circulating blood against "
                "artery walls. It is recorded using two numbers, "
                "such as 120/80 mmHg."
            ),
            "Cholesterol": (
                "A waxy substance found in the blood. The body "
                "needs cholesterol, but unhealthy levels can "
                "contribute to plaque buildup."
            ),
            "Chest Pain (Angina)": (
                "Discomfort or pain that can occur when the "
                "heart muscle does not receive enough blood."
            ),
            "Resting ECG": (
                "An electrocardiogram records the electrical "
                "activity of the heart while a person is at rest."
            ),
            "Maximum Heart Rate": (
                "The maximum heart rate recorded during the "
                "exercise test used in this dataset."
            ),
            "Exercise-Induced Angina": (
                "Chest discomfort reported during physical "
                "exercise or exertion."
            ),
            "ST Depression (Oldpeak)": (
                "A measurement of ST-segment depression during "
                "exercise relative to rest in the original dataset."
            ),
            "Slope of ST Segment": (
                "Describes the direction of the ST segment "
                "during peak exercise in the original dataset."
            ),
            "Thal (Dataset Code)": (
                "A dataset-specific coded variable historically "
                "associated with thalassemia-related test findings."
            ),
            "Major Vessels (CA)": (
                "The number of major vessels colored by "
                "fluoroscopy in the original dataset."
            ),
            "Fasting Blood Sugar": (
                "The blood glucose measurement after fasting. "
                "The original dataset includes an indicator "
                "for values above 120 mg/dl."
            )
        }

        for term, description in terms.items():
            with st.expander(term):
                st.write(description)

    elif topic == topics[4]:

        section_header("Heart Disease Risk Factors")

        st.write("""
        Several factors can influence cardiovascular health.
        Some cannot be changed, while others may be improved
        through lifestyle changes and medical care.
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="custom-card">
                <h3>Non-Modifiable Factors</h3>
                <p>
                • Age<br>
                • Family history<br>
                • Genetic factors
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="custom-card">
                <h3>Modifiable Factors</h3>
                <p>
                • Smoking and tobacco use<br>
                • High blood pressure<br>
                • Unhealthy cholesterol levels<br>
                • Diabetes<br>
                • Physical inactivity
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.info(
            "Risk factors do not guarantee that a person "
            "will develop heart disease."
        )

    elif topic == topics[5]:

        section_header("Prevention and Heart Care")

        practices = [
            ("Balanced Diet",
             "Include fruits, vegetables, whole grains and balanced meals."),
            ("Regular Physical Activity",
             "Stay physically active according to your health condition and medical advice."),
            ("Avoid Tobacco",
             "Avoid smoking and other tobacco products."),
            ("Monitor Health",
             "Get appropriate blood pressure, cholesterol and blood sugar checks."),
            ("Healthy Sleep",
             "Maintain a consistent and sufficient sleep routine."),
            ("Medical Guidance",
             "Follow advice from qualified healthcare professionals.")
        ]

        for title, description in practices:
            st.markdown(f"""
            <div class="custom-card">
                <h3>{title}</h3>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)

    elif topic == topics[6]:

        section_header("Understanding Our 13 Prediction Features")

        st.write("""
        Our machine learning model uses 13 input features
        from the Cleveland Heart Disease dataset.
        """)

        feature_data = [
            ["age", "Age", "Years"],
            ["sex", "Sex", "Encoded category (0 or 1)"],
            ["cp", "Chest Pain Type", "Encoded category (1 to 4)"],
            ["trestbps", "Resting Blood Pressure", "mmHg"],
            ["chol", "Serum Cholesterol", "mg/dl"],
            ["fbs", "Fasting Blood Sugar", "Encoded indicator (0 or 1)"],
            ["restecg", "Resting ECG", "Encoded category (0 to 2)"],
            ["thalach", "Maximum Heart Rate", "Beats per minute"],
            ["exang", "Exercise-Induced Angina", "Encoded indicator (0 or 1)"],
            ["oldpeak", "ST Depression", "Dataset measurement"],
            ["slope", "ST Segment Slope", "Encoded category (1 to 3)"],
            ["ca", "Major Vessels", "Count (0 to 3)"],
            ["thal", "Thal", "Encoded category (3, 6 or 7)"]
        ]

        feature_df = pd.DataFrame(
            feature_data,
            columns=[
                "Feature Name",
                "Meaning",
                "Representation"
            ]
        )

        st.dataframe(
            feature_df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Understanding Encoded Values")

        st.write("""
        Some dataset features are stored as numbers representing
        categories rather than direct measurements.

        For example:

        - Chest pain type uses values 1, 2, 3 and 4.
        - Resting ECG uses values 0, 1 and 2.
        - Exercise-induced angina uses 0 and 1.

        These are codes used by the original dataset.
        They should not be interpreted as severity scores.
        """)

        st.warning(
            "The model is trained on these particular dataset "
            "encodings. The prediction form is an academic "
            "demonstration, not a substitute for clinical testing."
        )

    st.markdown("---")

    st.caption(
        "Educational information only. Consult qualified "
        "healthcare professionals for personal medical concerns."
    )


# =====================================================
# HEART DISEASE PREDICTION PAGE
# =====================================================

elif page == "Heart Disease Prediction":

    st.markdown("""
    <div class="hero">
        <h1>🤖 Heart Disease Prediction</h1>
        <p>
        Enter sample dataset values to explore the output
        generated by our trained machine learning model.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.warning(
        "Educational demonstration only. This model is not a "
        "medical diagnostic tool. Do not use its output to make "
        "healthcare decisions."
    )

    with st.form("prediction_form"):

        section_header("Enter Input Information")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### Personal & Clinical Details")

            age = st.number_input(
                "Age (years)",
                min_value=1,
                max_value=120,
                value=45
            )

            sex = st.selectbox(
                "Sex",
                [0, 1],
                format_func=lambda x:
                "Female (0)" if x == 0 else "Male (1)"
            )

            trestbps = st.number_input(
                "Resting Blood Pressure (mmHg)",
                min_value=50,
                max_value=250,
                value=120
            )

            chol = st.number_input(
                "Cholesterol (mg/dl)",
                min_value=50,
                max_value=700,
                value=200
            )

            fbs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl",
                [0, 1]
            )

            thalach = st.number_input(
                "Maximum Heart Rate (bpm)",
                min_value=50,
                max_value=250,
                value=150
            )

            exang = st.selectbox(
                "Exercise-Induced Angina",
                [0, 1]
            )

        with col2:

            st.markdown("### Additional Dataset Features")

            cp = st.selectbox(
                "Chest Pain Type",
                [1, 2, 3, 4]
            )

            restecg = st.selectbox(
                "Resting ECG",
                [0, 1, 2]
            )

            oldpeak = st.number_input(
                "ST Depression (Oldpeak)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1
            )

            slope = st.selectbox(
                "Slope of ST Segment",
                [1, 2, 3]
            )

            ca = st.selectbox(
                "Number of Major Vessels",
                [0, 1, 2, 3]
            )

            thal = st.selectbox(
                "Thal Dataset Code",
                [3, 6, 7]
            )

        submit = st.form_submit_button(
            "Generate Model Output",
            use_container_width=True
        )

    if submit:

        input_data = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }])

        prediction = int(model.predict(input_data)[0])

        probability = float(
            model.predict_proba(input_data)[0][1]
        )

        database.save_prediction(
            age=age,
            sex=sex,
            chest_pain=cp,
            cholesterol=chol,
            blood_pressure=trestbps,
            predicted_class=prediction,
            probability=probability
        )

        st.markdown("---")

        section_header("Model Output")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Dataset Class",
                str(prediction)
            )

        with col2:
            st.metric(
                "Model Probability for Class 1",
                f"{probability * 100:.2f}%"
            )

        if prediction == 1:
            st.error(
                "The model predicted class 1 "
                "(heart disease present in the dataset's coding)."
            )
        else:
            st.success(
                "The model predicted class 0 "
                "(heart disease absent in the dataset's coding)."
            )

        st.success("Prediction record saved successfully!")

        st.caption(
            "The displayed probability is the model's output "
            "for class 1. It is not an individual's clinical "
            "probability or a medical diagnosis."
        )


# =====================================================
# MODEL PERFORMANCE PAGE
# =====================================================

elif page == "Model Performance":

    st.markdown("""
    <div class="hero">
        <h1>📊 Model Performance</h1>
        <p>
        Evaluate and compare the three machine learning
        algorithms used in this project.
        </p>
    </div>
    """, unsafe_allow_html=True)

    section_header("Model Comparison")

    display_results = results.copy()

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]

    for column in metric_columns:
        display_results[column] = (
            display_results[column] * 100
        ).round(2)

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    section_header("Accuracy and F1 Score Comparison")

    chart_data = results.set_index("Model")[
        ["Accuracy", "F1 Score"]
    ] * 100

    st.bar_chart(chart_data)

    st.markdown("---")

    section_header("Detailed Model Evaluation")

    model_options = {
        "Logistic Regression": "logistic_regression_evaluation.pkl",
        "Decision Tree": "decision_tree_evaluation.pkl",
        "Random Forest": "random_forest_evaluation.pkl"
    }

    selected_model = st.selectbox(
        "Select a model",
        list(model_options.keys())
    )

    evaluation = joblib.load(
        model_options[selected_model]
    )

    cm = evaluation["confusion_matrix"]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "ROC-AUC Score",
            f"{evaluation['roc_auc'] * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Correct Predictions",
            cm[0][0] + cm[1][1]
        )

    st.markdown("---")

    section_header("Confusion Matrix")

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Class 0", "Class 1"],
        yticklabels=["Class 0", "Class 1"],
        linewidths=1,
        linecolor="#15283D",
        ax=ax
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title(selected_model + " Confusion Matrix")

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Class 0 and Class 1 represent the encoded "
        "target categories in the dataset."
    )

    st.markdown("---")

    section_header("Classification Report")

    report = pd.DataFrame(
        evaluation["classification_report"]
    ).transpose()

    st.dataframe(
        report.round(3),
        use_container_width=True
    )

    st.markdown("---")

    section_header("ROC Curve Comparison")

    fig, ax = plt.subplots(figsize=(9, 5))

    for name, filename in model_options.items():

        roc_data = joblib.load(filename)

        ax.plot(
            roc_data["fpr"],
            roc_data["tpr"],
            linewidth=2,
            label=f"{name} (AUC = {roc_data['roc_auc']:.3f})"
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "ROC curves compare true positive and false positive "
        "rates across classification thresholds."
    )

    st.info(
        "Evaluation results are based on a single held-out test "
        "split of 60 records. They do not establish clinical validity."
    )


# =====================================================
# DATASET ANALYSIS PAGE
# =====================================================

elif page == "Dataset Analysis":

    st.markdown("""
    <div class="hero">
        <h1>🔬 Dataset Analysis</h1>
        <p>
        Explore the Cleveland Heart Disease dataset
        used to train and evaluate our models.
        </p>
    </div>
    """, unsafe_allow_html=True)

    section_header("Dataset Preview")

    st.dataframe(
        data.head(10),
        use_container_width=True
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", len(data))

    with col2:
        st.metric("Input Features", len(data.columns) - 1)

    with col3:
        st.metric("Target Classes", data["target"].nunique())

    st.markdown("---")

    section_header("Target Distribution")

    target_counts = data["target"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.barplot(
        x=target_counts.index.astype(str),
        y=target_counts.values,
        ax=ax
    )

    ax.set_xlabel("Target Class")
    ax.set_ylabel("Number of Records")
    ax.set_title("Heart Disease Target Distribution")

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    section_header("Statistical Summary")

    st.dataframe(
        data.describe().round(2),
        use_container_width=True
    )


# =====================================================
# PREDICTION HISTORY PAGE
# =====================================================

elif page == "Prediction History":

    st.markdown("""
    <div class="hero">
        <h1>🗂 Prediction History</h1>
        <p>
        View and download prediction records stored
        in the local SQLite database.
        </p>
    </div>
    """, unsafe_allow_html=True)

    history = database.get_predictions()

    if history.empty:

        st.info(
            "No prediction records found. "
            "Make a prediction first to see it here."
        )

    else:

        section_header("Prediction Statistics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Predictions", len(history))

        with col2:
            st.metric(
                "Class 1 Predictions",
                int((history["predicted_class"] == 1).sum())
            )

        with col3:
            st.metric(
                "Class 0 Predictions",
                int((history["predicted_class"] == 0).sum())
            )

        st.markdown("---")

        section_header("Saved Prediction Records")

        display_history = history.copy()

        display_history["predicted_class"] = (
            display_history["predicted_class"].map({
                0: "Class 0",
                1: "Class 1"
            })
        )

        display_history["probability"] = (
            display_history["probability"] * 100
        ).round(2).astype(str) + "%"

        st.dataframe(
            display_history,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        csv_data = history.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇ Download Prediction History (CSV)",
            data=csv_data,
            file_name="heart_prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.caption(
            "Records are stored locally in SQLite. "
            "Model outputs are not medical diagnoses."
        )


# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown("""
<div class="footer">
    <b style="color:#45D9F5;">❤️ HeartCare AI</b><br>
    Heart Disease Prediction Using Machine Learning<br>
    Academic Mini Project | Educational Use Only
</div>
""", unsafe_allow_html=True)
