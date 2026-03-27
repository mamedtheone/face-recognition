from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.utils import get_project_status, register_face_sample, train_model_from_registered_faces  # noqa: E402

st.set_page_config(page_title="Register Faces")
st.title("Register User Faces")
st.write("Capture multiple face samples per user. A minimum of 5 samples is recommended.")

user_name = st.text_input("Name or ID")
photo = st.camera_input("Take a registration photo")

if st.button("Save face sample", use_container_width=True):
    if not user_name.strip():
        st.warning("Enter a name or ID before saving.")
    elif photo is None:
        st.warning("Capture a photo first.")
    else:
        result = register_face_sample(user_name, photo.getvalue())
        st.success(
            f"Saved sample for {result['user_id']}. Processed samples: {result['processed_count']}. "
            f"Recommended samples remaining: {result['remaining_recommended_samples']}."
        )

st.divider()
st.subheader("Train Model")
st.write("Train or retrain the KNN model after collecting enough samples for at least two users.")

if st.button("Train / Retrain Model", use_container_width=True):
    try:
        artifact = train_model_from_registered_faces()
        st.success(
            f"Training complete. Accuracy: {artifact['metrics']['accuracy']:.2%}. "
            f"Distance threshold: {artifact['distance_threshold']:.3f}"
        )
        st.dataframe(artifact["metrics"]["confusion_matrix"], use_container_width=True)
    except Exception as error:
        st.error(str(error))

status = get_project_status()
if status["users"]:
    st.subheader("Current Dataset")
    st.dataframe(status["users"], use_container_width=True)
