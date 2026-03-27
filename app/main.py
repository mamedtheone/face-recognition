from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.utils import get_project_status, training_readiness_message  # noqa: E402

st.set_page_config(page_title="FaceAuth AI", layout="wide")

status = get_project_status()

st.title("FaceAuth: AI Face Login System")
st.write(
    "This Streamlit app lets users register with face images, train a KNN classifier, and authenticate with facial recognition."
)

col1, col2, col3 = st.columns(3)
col1.metric("Registered Users", status["registered_user_count"])
col2.metric("Users Ready for Training", status["trainable_users"])
col3.metric("Model Available", "Yes" if status["model_ready"] else "No")

st.subheader("Workflow")
st.markdown(
    """
1. Open the **Register** page and capture 5 to 10 face samples for each user.
2. Train or retrain the model once at least two users have enough processed samples.
3. Open the **Login** page and authenticate using a fresh face image.
"""
)

st.info(training_readiness_message())

if status["metrics"]:
    st.subheader("Latest Evaluation")
    st.write(f"Accuracy: **{status['metrics'].get('accuracy', 0.0):.2%}**")
    st.write("Confusion Matrix")
    st.dataframe(status["metrics"].get("confusion_matrix", []), use_container_width=True)

if status["users"]:
    st.subheader("Registered Users")
    st.dataframe(status["users"], use_container_width=True)
else:
    st.warning("No users have been registered yet.")
