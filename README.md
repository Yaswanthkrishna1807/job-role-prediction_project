# Job Role Predictor

Streamlit app that predicts a job role from skills, qualification, and experience level. It trains five classifiers when the app starts and lets the user choose one.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Keep `jobs_skills_dataset.csv` beside `app.py`.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py`, `requirements.txt`, and `jobs_skills_dataset.csv`.
2. In Streamlit Community Cloud, create an app from that repository and set the entrypoint to `app.py`.
3. Deploy.

Review the CSV before publishing: a public repository makes its files accessible to others.
