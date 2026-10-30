import html
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


st.set_page_config(
    page_title="Career Compass | Job Role Predictor",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --ink: #17263c;
        --muted: #68788e;
        --teal: #087f8c;
        --line: #e2e9f0;
    }

    .stApp {
        background: linear-gradient(180deg, #f4f8fb 0%, #f8fafc 48%, #f4f8fb 100%);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1180px;
        padding-top: 2.1rem;
        padding-bottom: 3rem;
    }

    html, body, [class*="css"] {
        font-family: "DM Sans", sans-serif;
    }

    h1, h2, h3 {
        font-family: "Manrope", sans-serif !important;
        color: var(--ink);
        letter-spacing: -0.025em;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.25rem 2.5rem;
        border-radius: 24px;
        background: linear-gradient(115deg, #102b46 0%, #17435a 58%, #087f8c 100%);
        color: white;
        box-shadow: 0 18px 45px rgba(20, 53, 78, 0.14);
        margin-bottom: 1.6rem;
    }

    .hero:after {
        content: "🧭";
        position: absolute;
        right: 3rem;
        top: 50%;
        transform: translateY(-50%);
        font-size: 5rem;
        opacity: 0.16;
    }

    .hero .eyebrow {
        color: #92e3d7;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 0.73rem;
        font-weight: 700;
    }

    .hero h1 {
        color: #fff;
        margin: 0.55rem 0 0.45rem;
        font-size: 2.25rem;
    }

    .hero p {
        color: #d5e4ed;
        margin: 0;
        font-size: 1rem;
        max-width: 670px;
    }

    .section-kicker {
        color: var(--teal);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.72rem;
        font-weight: 700;
    }

    [data-testid="stForm"] {
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 1.5rem 1.65rem;
        box-shadow: 0 10px 30px rgba(25, 52, 78, 0.055);
    }

    [data-testid="stWidgetLabel"] p {
        font-weight: 600;
        color: #263c53;
    }

    [data-testid="stFormSubmitButton"] button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 2.8rem;
        border: 0;
        background: linear-gradient(100deg, #087f8c, #0a9a91);
    }

    [data-testid="stSidebar"] {
        background: #102b46;
    }

    [data-testid="stSidebar"] * {
        color: #eaf2f7;
    }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #fff;
    }

    .result-role {
        font-family: "Manrope", sans-serif;
        font-size: 1.85rem;
        line-height: 1.2;
        font-weight: 800;
        color: #153149;
    }

    .result-label {
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-size: 0.7rem;
        font-weight: 700;
    }

    .skill-chip {
        display: inline-block;
        background: #e6f5f3;
        border: 1px solid #c8eae5;
        color: #096c73;
        padding: 0.42rem 0.72rem;
        border-radius: 999px;
        margin: 0.2rem 0.25rem 0.2rem 0;
        font-size: 0.88rem;
        font-weight: 600;
    }

    .small-note {
        color: var(--muted);
        font-size: 0.84rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Career Compass · Skills based guidance</div>
      <h1>Find your next career direction</h1>
      <p>
        Match your skills, education, and experience to a role,
        then see practical skills to build next.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
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
        df["experience_level"],
        errors="coerce",
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
            skill
            for skill, _ in Counter(all_skills).most_common(15)
        ]

    # Create dropdown choices from the dataset.
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


with st.sidebar:
    st.markdown("## 🧭 Career Compass")
    st.markdown("A focused role matcher powered by the skills dataset.")
    st.markdown("---")
    st.markdown("### How to use it")
    st.markdown(
        "1. Select every skill that applies.\n"
        "2. Choose your qualification and experience.\n"
        "3. Pick a model and submit."
    )
    st.markdown("---")
    st.caption(
        f"{len(df):,} cleaned records · "
        f"{df['job_role'].nunique()} role categories"
    )
    st.caption(
        "For exploration and career planning; predictions depend on "
        "the supplied dataset."
    )


st.markdown(
    '<div class="section-kicker">Your profile</div>',
    unsafe_allow_html=True,
)
st.markdown("### Tell us what you bring")
st.markdown(
    '<p class="small-note">'
    "Dropdown options are drawn from the dataset. Select multiple skills."
    "</p>",
    unsafe_allow_html=True,
)

with st.form("candidate_details"):
    skills = st.multiselect(
        "Your skills (select all that apply)",
        options=skill_options,
        help="Options are taken from the dataset. You can select multiple skills.",
        placeholder="Search and choose your skills",
    )

    col_qualification, col_experience = st.columns(2)

    with col_qualification:
        qualification = st.selectbox(
            "Qualification",
            options=qualification_options,
        )

    with col_experience:
        experience = st.selectbox(
            "Experience level · years",
            options=experience_options,
            format_func=lambda value: (
                str(int(value))
                if float(value).is_integer()
                else str(value)
            ),
        )

    col_model, col_action = st.columns([1, 1])

    with col_model:
        model_name = st.selectbox(
            "Prediction model",
            options=list(models.keys()),
            index=1,
        )

    with col_action:
        st.markdown(
            "<div style='height:1.82rem'></div>",
            unsafe_allow_html=True,
        )
        submitted = st.form_submit_button(
            "✨  Find my role",
            use_container_width=True,
        )


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
            skill.strip().lower()
            for skill in skills
        }

        suggestions = [
            skill
            for skill in role_skill_map.get(predicted_role, [])
            if skill.lower() not in selected_skill_set
        ][:5]

        st.markdown(
            '<div class="section-kicker">Your match</div>',
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            left, right = st.columns([1.3, 1])

            with left:
                st.markdown(
                    '<div class="result-label">Suggested role</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="result-role">'
                    f'{html.escape(predicted_role)}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<p class="small-note">Predicted with '
                    f'{html.escape(model_name)}</p>',
                    unsafe_allow_html=True,
                )

            with right:
                st.markdown(
                    '<div class="result-label">'
                    "Next skills to explore"
                    "</div>",
                    unsafe_allow_html=True,
                )

                if suggestions:
                    chips = "".join(
                        f'<span class="skill-chip">'
                        f'{html.escape(skill)}</span>'
                        for skill in suggestions
                    )
                    st.markdown(chips, unsafe_allow_html=True)
                else:
                    st.info(
                        "You already selected the common skills listed "
                        "for this role."
                    )


with st.expander("About this app"):
    st.write(
        f"Trained from {len(df):,} records using skills, qualification, "
        "and experience level."
    )
    st.caption(
        "Predictions are based on the supplied dataset and are for guidance only."
    )
