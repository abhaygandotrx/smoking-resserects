# app.py
import streamlit as st
import cv2
import numpy as np
import os
import time
from detector import LungAssistantDetector

st.set_page_config(page_title="Smoking Risk Detector", layout="wide")
st.title("Smoking Detection and Health Risk Estimator")

# Load detector (auto-fallback)
model_file = "best.pt"
if os.path.exists(model_file):
    detector = LungAssistantDetector(model_path=model_file)
    st.info(f"Loaded model: {model_file}")
else:
    detector = LungAssistantDetector(model_path=None)
    st.info("No custom model found — using default yolov8n (heuristic detection).")

# Sidebar health inputs
st.sidebar.header("Personal Inputs")
age = st.sidebar.number_input("Age", 18, 90, 30)
years_smoked = st.sidebar.number_input("Years smoked", 0, 50, 5)
cigs_per_day = st.sidebar.number_input("Cigarettes/day", 0, 60, 10)
nutrition = st.sidebar.selectbox("Nutrition", ["Poor", "Average", "Good"])
fitness = st.sidebar.selectbox("Physical activity", ["Low", "Moderate", "High"])

def calculate_life_expectancy(years_smoked, cigs_per_day, nutrition, fitness):
    base = 80.0
    loss = years_smoked * 0.25 + cigs_per_day * 0.12
    if nutrition == "Poor":
        loss += 2.5
    elif nutrition == "Average":
        loss += 1.0
    if fitness == "Low":
        loss += 3.0
    elif fitness == "Moderate":
        loss += 1.0
    life = max(20.0, base - loss)
    risk_pct = min(99.0, loss * 1.2)
    return risk_pct, life

risk_pct, life_years = calculate_life_expectancy(years_smoked, cigs_per_day, nutrition, fitness)

st.subheader("Estimated health")
st.write(f"Estimated risk: {risk_pct:.1f}%")
st.write(f"Estimated life expectancy (years): {life_years:.1f}")

# Webcam controls
if 'run_webcam' not in st.session_state:
    st.session_state.run_webcam = False

col1, col2 = st.columns(2)
if col1.button("Start Webcam", key="start_cam"):
    st.session_state.run_webcam = True
if col2.button("Stop Webcam", key="stop_cam"):
    st.session_state.run_webcam = False

frame_placeholder = st.empty()
status_placeholder = st.empty()

cap = None
if st.session_state.run_webcam:
    cap = cv2.VideoCapture(0)
    time.sleep(0.5)

# Run loop (Streamlit-friendly: use session_state flag)
while st.session_state.run_webcam:
    ret, frame_bgr = cap.read()
    if not ret:
        status_placeholder.error("Cannot access webcam.")
        break

    # convert to RGB for detector
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    smoking_flag, detections = detector.detect_smoking(frame_rgb)

    # draw boxes
    for name, conf, (x1, y1, x2, y2) in detections:
        color = (0, 255, 0) if name == "person" else (255, 0, 0)
        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame_rgb, f"{name} {conf:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # convert to displayable image
    frame_placeholder.image(frame_rgb, channels="RGB")

    if smoking_flag:
        status_placeholder.warning("Smoking detected — risk increases.")
    else:
        status_placeholder.info("No smoking detected.")

    # small sleep so UI doesn't hog CPU fully
    time.sleep(0.05)

if cap:
    cap.release()
