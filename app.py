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
st.write(
    "Enter your skills, qualification, and experience to get a suggested role "
    "and skills to build."
)

DATA_PATH = Path(__file__).parent / "jobs_skills_dataset.csv"


@st.cache_resource
def train_models():
    df = pd.read_csv(DATA_PATH)

    required = ["skills", "qualification", "experience_level", "job_role"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df.dropna(subset=required).drop_duplicates().reset_index(drop=True)

    for column in ["skills", "qualification", "job_role"]:
        df[column] = df[column].astype(str).str.strip()

    df["experience_level"] = pd.to_numeric(
        df["experience_level"], errors="coerce"
    )
    df = df.dropna(subset=["experience_level"]).reset_index(drop=True)

    def make_features(data):
        return (
            "skills: "
            + data["skills"].fillna("").astype(str)
            + " qualification: "
            + data["qualification"].fillna("").astype(str)
            + " experience: "
            + data["experience_level"].fillna("").astype(str)
        )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=1000,
    )
    X = vectorizer.fit_transform(make_features(df)).toarray()

    encoder = LabelEncoder()
    y = encoder.fit_transform(df["job_role"])

    class_counts = Counter(y)
    if min(class_counts.values()) < 2:
        raise ValueError(
            "SMOTE requires at least two records for every job role."
        )

    neighbors = min(3, min(class_counts.values()) - 1)
    X_balanced, y_balanced = SMOTE(
        random_state=42,
        k_neighbors=neighbors,
    ).fit_resample(X, y)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "SVM": LinearSVC(),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            eval_metric="mlogloss",
            random_state=42,
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2),
            n_estimators=200,
            learning_rate=0.5,
            random_state=42,
        ),
    }

    for model in models.values():
        model.fit(X_balanced, y_balanced)

    role_skill_map = {}
    for role, group in df.groupby("job_role"):
        all_skills = (
            group["skills"]
            .str.lower()
            .str.split(",")
            .explode()
            .str.strip()
        )
        role_skill_map[role] = [
            skill for skill, _ in Counter(all_skills).most_common(15)
        ]

    # Build dropdown choices from the dataset.
    skill_options = sorted(
        {
            skill.strip()
            for value in df["skills"]
            for skill in str(value).split(",")
            if skill.strip()
        },
        key=str.casefold,
    )
    qualification_options = sorted(
        df["qualification"].dropna().unique(),
        key=str.casefold,
    )
    experience_options = sorted(
        df["experience_level"].dropna().unique()
    )

    return (
        df,
        vectorizer,
        encoder,
        models,
        role_skill_map,
        make_features,
        skill_options,
        qualification_options,
        experience_options,
    )


try:
    (
        df,
        vectorizer,
        encoder,
        models,
        role_skill_map,
        make_features,
        skill_options,
        qualification_options,
        experience_options,
    ) = train_models()
except Exception as exc:
    st.error(f"Could not load or train the model: {exc}")
    st.stop()


with st.form("candidate_details"):
    skills = st.multiselect(
        "Your skills (select all that apply)",
        options=skill_options,
        help="Options are taken from the dataset. You can select multiple skills.",
    )

    qualification = st.selectbox(
        "Your qualification",
        options=qualification_options,
    )

    experience = st.selectbox(
        "Experience level (years)",
        options=experience_options,
        format_func=lambda value: (
            str(int(value))
            if float(value).is_integer()
            else str(value)
        ),
    )

    model_name = st.selectbox(
        "Model",
        options=list(models.keys()),
        index=1,
    )

    submitted = st.form_submit_button("Predict role")


if submitted:
    if not skills:
        st.warning("Please select at least one skill.")
    else:
        selected_skills = ", ".join(skills)

        candidate = pd.DataFrame(
            [
                {
                    "skills": selected_skills,
                    "qualification": qualification,
                    "experience_level": experience,
                }
            ]
        )

        vector = vectorizer.transform(make_features(candidate))
        predicted_label = models[model_name].predict(vector)[0]
        predicted_role = encoder.inverse_transform([predicted_label])[0]

        selected_skill_set = {
            skill.strip().lower() for skill in skills
        }
        suggestions = [
            skill
            for skill in role_skill_map.get(predicted_role, [])
            if skill.lower() not in selected_skill_set
        ][:5]

        st.success(f"Predicted job role: **{predicted_role}**")
        st.subheader("Skills to consider building")

        if suggestions:
            for skill in suggestions:
                st.write(f"- {skill}")
        else:
            st.write(
                "You already have the common skills listed for this role."
            )


with st.expander("About this app"):
    st.write(
        f"Trained from {len(df):,} records using skills, qualification, "
        "and experience level."
    )
    st.caption(
        "Predictions are based on the supplied dataset and are for guidance only."
    )
