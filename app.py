# app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# 1) Page config (must be first Streamlit call)
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

hide_streamlit_style = """
            <style>
            /* Hide the Streamlit "hamburger" menu */
            #MainMenu { display: none; }

            /* Hide the Streamlit footer using a more specific selector */
            section[data-testid="stAppViewContainer"] footer { display: none !important; }

            /* Optional: Hide the header */
            /* header[data-testid="stHeader"] { display: none !important; } */

            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# 2) Load processed data for modeling
@st.cache_data
def load_model_data():
    df = pd.read_csv("Selected_Features_data_processed.csv")
    df["FraudFound"] = pd.to_numeric(df["FraudFound"], errors="coerce")
    return df.dropna(subset=["FraudFound"])

df_model = load_model_data()
X = df_model.drop(columns=["FraudFound"])
y = df_model["FraudFound"]

# 3) Load raw data for risk panel (and coerce FraudFound → numeric)
@st.cache_data
def load_raw_data():
    df = pd.read_csv("feature_selected_dataset.csv")
    # if your raw uses "Yes"/"No", map it here. Otherwise to_numeric will convert "1"/"0" strings
    df["FraudFound"] = df["FraudFound"].map({"No": 0, "Yes": 1})
    # fallback: ensure numeric
    df["FraudFound"] = pd.to_numeric(df["FraudFound"], errors="coerce")
    return df.dropna(subset=["FraudFound"])

df_raw = load_raw_data()

# 4) Define your models + hyperparameter grids (with random_state=24)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=24),
    "Random Forest": RandomForestClassifier(random_state=24),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(),
    "ANN": MLPClassifier(max_iter=500, random_state=24),
    "SVM (Linear)": SVC(kernel="linear", probability=True, random_state=24),
    "SVM (RBF)": SVC(kernel="rbf",      probability=True, random_state=24),
    "Logistic Reg.": LogisticRegression(solver="liblinear", random_state=24)
}

param_grids = {
    "Decision Tree":  {"max_depth": [3,5,10,None],       "min_samples_split": [2,5,10]},
    "Random Forest":  {"n_estimators": [50,100],          "max_depth": [5,10,None]},
    "Naive Bayes":    {"var_smoothing": [1e-9,1e-8,1e-7]},
    "KNN":            {"n_neighbors": [3,5,7,10],        "metric": ["minkowski","euclidean"]},
    "ANN":           {"hidden_layer_sizes":[(50,),(100,),(50,50)],
                       "activation":["relu","tanh"],       "solver":["adam"]},
    "SVM (Linear)":   {"C": [0.1,1,10]},
    "SVM (RBF)":      {"C": [0.1,1,10],                    "gamma":[0.01,0.1,1]},
    "Logistic Reg.":  {"C":[0.1,1,10]}
}

# 5) Session state for “has searched?”
if "ran_search" not in st.session_state:
    st.session_state.ran_search = False

# 6) Sidebar: Model Configuration
st.sidebar.header("⚙️ Model Configuration")
model_name = st.sidebar.selectbox("Select model", list(models.keys()))
use_smote  = st.sidebar.checkbox("Apply SMOTE", value=True)
cv_folds   = st.sidebar.slider("CV folds", 3, 10, 5)
if st.sidebar.button("Run Grid Search"):
    st.session_state.ran_search = True

# 7) Main: run train/test/grid & show metrics once triggered
if st.session_state.ran_search:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=24, stratify=y
    )
    if use_smote:
        X_train, y_train = SMOTE(random_state=24).fit_resample(X_train, y_train)

    grid = GridSearchCV(
        models[model_name],
        param_grids[model_name],
        cv=cv_folds,
        scoring="f1_macro",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    y_pred = grid.best_estimator_.predict(X_test)

    st.subheader(f"🏆 Best params for {model_name}")
    st.write(grid.best_params_)
    st.subheader("📊 Metrics")
    st.write({
        "Accuracy": accuracy_score(y_test, y_pred),
        "Macro F1": f1_score(y_test, y_pred, average="macro")
    })
    st.subheader("📝 Classification Report")
    st.text(classification_report(y_test, y_pred))
    st.subheader("📈 Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

    # 8) Unlock Risk Drill-Down
    st.sidebar.header("🔍 Risk Drill-Down")
    raw_feats = [
        "Sex","Fault","PolicyType","VehiclePrice","PastNumberOfClaims",
        "AgeOfVehicle","AgeOfPolicyHolder","PoliceReportFiled",
        "WitnessPresent","AgentType","BasePolicy"
    ]
    feat = st.sidebar.selectbox("Choose feature", raw_feats)
    if feat:
        risk = (
            df_raw
            .groupby(feat)["FraudFound"]
            .mean()
            .reset_index()
            .rename(columns={"FraudFound":"Fraud Probability"})
        )
        risk["Fraud Probability"] = risk["Fraud Probability"].fillna(0)
        st.subheader(f"🔎 Fraud Probability by {feat}")
        st.dataframe(risk.style.format({"Fraud Probability":"{:.1%}"}))
        st.bar_chart(risk.set_index(feat)["Fraud Probability"])

else:
    st.info("Select your model and click **Run Grid Search** to begin.")
