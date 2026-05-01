from __future__ import annotations

import json

import requests
import streamlit as st

API_ROOT = st.secrets.get("api_root", "http://localhost:8000/api")
HEADERS = {"X-User-Id": "dev_user"}

st.set_page_config(page_title="DuckDB Predictive Agent", layout="wide")
st.title("DuckDB Predictive Agent")

with st.sidebar:
    st.header("Upload Database")
    uploaded = st.file_uploader("DuckDB File", type=["duckdb"])
    display_name = st.text_input("Display Name", value="Demo Database")
    if st.button("Upload And Connect", disabled=uploaded is None):
        files = {"file": (uploaded.name, uploaded.getvalue(), "application/octet-stream")}
        upload_resp = requests.post(f"{API_ROOT}/databases/uploads", files=files, headers=HEADERS, timeout=60)
        upload_resp.raise_for_status()
        upload_id = upload_resp.json()["upload_id"]
        connect_resp = requests.post(
            f"{API_ROOT}/databases/connect",
            json={"display_name": display_name, "upload_id": upload_id},
            headers=HEADERS,
            timeout=120,
        )
        connect_resp.raise_for_status()
        st.session_state["database_id"] = connect_resp.json()["database_id"]

database_id = st.text_input("Database ID", value=st.session_state.get("database_id", ""))

col1, col2 = st.columns(2)

with col1:
    st.subheader("Database Summary")
    if st.button("Load Summary", disabled=not database_id):
        response = requests.get(f"{API_ROOT}/databases/{database_id}", headers=HEADERS, timeout=60)
        if response.ok:
            st.session_state["summary"] = response.json()
        else:
            st.error(response.text)
    if "summary" in st.session_state:
        st.json(st.session_state["summary"])

    st.subheader("Ask Predictive Question")
    question = st.text_area("Question", value="Which customers are likely to churn next month?")
    if st.button("Run Prediction", disabled=not database_id):
        response = requests.post(
            f"{API_ROOT}/databases/{database_id}/predict",
            json={"question": question},
            headers=HEADERS,
            timeout=120,
        )
        if response.ok:
            st.session_state["prediction"] = response.json()
        else:
            st.error(response.text)
    if "prediction" in st.session_state:
        st.json(st.session_state["prediction"])

with col2:
    st.subheader("Profile")
    if st.button("Load Profile", disabled=not database_id):
        response = requests.get(f"{API_ROOT}/databases/{database_id}/profile", headers=HEADERS, timeout=120)
        if response.ok:
            st.session_state["profile"] = response.json()
        else:
            st.error(response.text)
    if "profile" in st.session_state:
        st.json(st.session_state["profile"])

    st.subheader("Query Logs")
    if st.button("Load Query Logs", disabled=not database_id):
        response = requests.get(
            f"{API_ROOT}/databases/{database_id}/query-logs",
            headers=HEADERS,
            timeout=60,
        )
        if response.ok:
            st.session_state["query_logs"] = response.json()
        else:
            st.error(response.text)
    if "query_logs" in st.session_state:
        st.json(st.session_state["query_logs"])

