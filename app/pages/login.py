from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.utils import authenticate_uploaded_face, get_project_status  # noqa: E402

st.set_page_config(page_title="Face Login")
st.title("Face Login")
st.write("Capture a fresh face image and let the model decide whether access should be granted.")

status = get_project_status()
if not status["model_ready"]:
    st.warning("Train the model before using the login page.")

photo = st.camera_input("Take a login photo")

if st.button("Authenticate", use_container_width=True):
    if photo is None:
        st.warning("Capture a login photo first.")
    else:
        try:
            result = authenticate_uploaded_face(photo.getvalue())
            if result["granted"]:
                st.success(result["message"])
            else:
                st.error(result["message"])

            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("Predicted User", result["predicted_label"])
            metric_col2.metric("Probability", f"{result['probability']:.2%}")
            metric_col3.metric("Distance", f"{result['distance']:.3f}")
            st.caption(f"Distance threshold: {result['distance_threshold']:.3f}")
        except Exception as error:
            st.error(str(error))
