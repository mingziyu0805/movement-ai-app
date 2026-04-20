import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go

# ================== 多语言 ==================
TEXTS = {
    "en": {
        "title": "Movement Assessment AI",
        "upload": "Upload Image",
        "analyze": "Analyze",
        "original": "Original Image",
        "result": "Result",
        "angles": "Joint Angles",
        "chart": "Chart",
        "no_pose": "No pose detected."
    },
    "zh": {
        "title": "运动评估AI",
        "upload": "上传图片",
        "analyze": "分析",
        "original": "原图",
        "result": "结果",
        "angles": "关节角度",
        "chart": "图表",
        "no_pose": "未检测到人体"
    }
}

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

def process_image(image, pose):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    try:
        results = pose.process(image_rgb)
    except:
        return image, {}

    angles = {}

    if not results or not results.pose_landmarks:
        return image, {}

    landmarks = results.pose_landmarks.landmark
    h, w, _ = image.shape

    mp_pose = mp.solutions.pose

    joints = {
        "left_knee": (mp_pose.PoseLandmark.LEFT_HIP,
                      mp_pose.PoseLandmark.LEFT_KNEE,
                      mp_pose.PoseLandmark.LEFT_ANKLE),

        "right_knee": (mp_pose.PoseLandmark.RIGHT_HIP,
                       mp_pose.PoseLandmark.RIGHT_KNEE,
                       mp_pose.PoseLandmark.RIGHT_ANKLE)
    }

    for name, (i, j, k) in joints.items():
        try:
            a = [landmarks[i].x*w, landmarks[i].y*h]
            b = [landmarks[j].x*w, landmarks[j].y*h]
            c = [landmarks[k].x*w, landmarks[k].y*h]
            angles[name] = calculate_angle(a, b, c)
        except:
            angles[name] = None

    mp.solutions.drawing_utils.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS
    )

    return image, angles


st.set_page_config(page_title="AI Movement", layout="wide")

lang = st.selectbox("Language", ["English", "中文"])
t = TEXTS["en" if lang == "English" else "zh"]

st.title(t["title"])

uploaded = st.file_uploader(t["upload"], type=["jpg", "png", "jpeg"])
run = st.button(t["analyze"])

@st.cache_resource
def load_model():
    return mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5
    )

if run and uploaded:
    pose = load_model()

    image = Image.open(uploaded).convert("RGB")
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption=t["original"], use_container_width=True)

    processed, angles = process_image(img, pose)

    with col2:
        st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), caption=t["result"], use_container_width=True)

    if angles:
        df = pd.DataFrame(list(angles.items()), columns=["Joint", "Angle"])
        st.dataframe(df)

        fig = go.Figure([go.Bar(x=df["Joint"], y=df["Angle"])])
        st.plotly_chart(fig)
    else:
        st.warning(t["no_pose"])
