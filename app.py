import pandas as pd
import streamlit as st
from collections import Counter
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

st.set_page_config(page_title="Job Role Predictor", page_icon="💼")
st.title("💼 Job Role Predictor")
st.write("Enter your skills, qualification, and experience to get a suggested role and skills to build.")
DATA_PATH = Path(__file__).parent / "jobs_skills_dataset.csv"

@st.cache_resource
def train_models():
    df = pd.read_csv(DATA_PATH)
    required = ["skills", "qualification", "experience_level", "job_role"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df.dropna(subset=required).drop_duplicates().reset_index(drop=True)
    for c in ["skills", "qualification", "job_role"]:
        df[c] = df[c].astype(str).str.strip()
    df["experience_level"] = pd.to_numeric(df["experience_level"], errors="coerce")
    df = df.dropna(subset=["experience_level"]).reset_index(drop=True)

    def make_features(data):
        return ("skills: " + data["skills"].fillna("").astype(str)
                + " qualification: " + data["qualification"].fillna("").astype(str)
                + " experience: " + data["experience_level"].fillna("").astype(str))

    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    X = vectorizer.fit_transform(make_features(df)).toarray()
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["job_role"])
    counts = Counter(y)
    if min(counts.values()) < 2:
        raise ValueError("SMOTE requires at least two records for every job role.")
    neighbors = min(3, min(counts.values()) - 1)
    X_bal, y_bal = SMOTE(random_state=42, k_neighbors=neighbors).fit_resample(X, y)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "SVM": LinearSVC(),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="mlogloss", random_state=42),
        "AdaBoost": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2),
                                       n_estimators=200, learning_rate=0.5, random_state=42),
    }
    for model in models.values():
        model.fit(X_bal, y_bal)

    role_skill_map = {}
    for role, group in df.groupby("job_role"):
        all_skills = group["skills"].str.lower().str.split(",").explode().str.strip()
        role_skill_map[role] = [s for s, _ in Counter(all_skills).most_common(15)]
    return df, vectorizer, encoder, models, role_skill_map, make_features

try:
    df, vectorizer, encoder, models, role_skill_map, make_features = train_models()
except Exception as exc:
    st.error(f"Could not load or train the model: {exc}")
    st.stop()

with st.form("candidate_details"):
    skills = st.text_area("Your skills (comma separated)", placeholder="Python, SQL, data analysis")
    qualification = st.text_input("Your qualification", placeholder="Bachelor's in Computer Science")
    experience = st.number_input("Experience level (years)", min_value=0.0, max_value=60.0, value=0.0, step=1.0)
    model_name = st.selectbox("Model", list(models.keys()), index=1)
    submitted = st.form_submit_button("Predict role")

if submitted:
    if not skills.strip() or not qualification.strip():
        st.warning("Please enter both skills and qualification.")
    else:
        row = pd.DataFrame([{"skills": skills, "qualification": qualification,
                             "experience_level": experience}])
        vector = vectorizer.transform(make_features(row))
        label = models[model_name].predict(vector)[0]
        role = encoder.inverse_transform([label])[0]
        user_skills = {s.strip().lower() for s in skills.split(",")}
        suggestions = [s for s in role_skill_map.get(role, []) if s.lower() not in user_skills][:5]
        st.success(f"Predicted job role: **{role}**")
        st.subheader("Skills to consider building")
        if suggestions:
            for skill in suggestions:
                st.write(f"- {skill}")
        else:
            st.write("You already have the common skills listed for this role.")

with st.expander("About this app"):
    st.write(f"Trained from {len(df):,} records using skills, qualification, and experience level.")
    st.caption("Predictions are based on the supplied dataset and are for guidance only.")
