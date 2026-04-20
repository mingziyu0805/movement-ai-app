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
        "title": "Movement Assessment & AI Platform",
        "upload": "Upload Image (JPG/PNG)",
        "analyze": "Analyze",
        "original": "Original Image",
        "result": "Analysis Result",
        "angles": "Joint Angles",
        "chart": "Angle Chart",
        "no_pose": "No human pose detected.",
        "left_knee": "Left Knee", "right_knee": "Right Knee",
        "left_hip": "Left Hip", "right_hip": "Right Hip",
        "left_ankle": "Left Ankle", "right_ankle": "Right Ankle",
        "left_elbow": "Left Elbow", "right_elbow": "Right Elbow",
        "left_shoulder": "Left Shoulder", "right_shoulder": "Right Shoulder"
    },
    "ko": {
        "title": "움직임 평가 AI 플랫폼",
        "upload": "이미지 업로드 (JPG/PNG)",
        "analyze": "분석",
        "original": "원본 이미지",
        "result": "분석 결과",
        "angles": "관절 각도",
        "chart": "각도 차트",
        "no_pose": "인체 자세가 감지되지 않았습니다.",
        "left_knee": "왼쪽 무릎", "right_knee": "오른쪽 무릎",
        "left_hip": "왼쪽 엉덩이", "right_hip": "오른쪽 엉덩이",
        "left_ankle": "왼쪽 발목", "right_ankle": "오른쪽 발목",
        "left_elbow": "왼쪽 팔꿈치", "right_elbow": "오른쪽 팔꿈치",
        "left_shoulder": "왼쪽 어깨", "right_shoulder": "오른쪽 어깨"
    },
    "zh": {
        "title": "运动评估AI平台",
        "upload": "上传图片 (JPG/PNG)",
        "analyze": "分析",
        "original": "原始图片",
        "result": "分析结果",
        "angles": "关节角度",
        "chart": "角度图表",
        "no_pose": "未检测到人体姿态。",
        "left_knee": "左膝", "right_knee": "右膝",
        "left_hip": "左髋", "right_hip": "右髋",
        "left_ankle": "左踝", "right_ankle": "右踝",
        "left_elbow": "左肘", "right_elbow": "右肘",
        "left_shoulder": "左肩", "right_shoulder": "右肩"
    }
}

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def process_image(image, pose):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    angles = {}

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        h, w, _ = image.shape

        joints = {
            'left_hip': (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.LEFT_KNEE),
            'right_hip': (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_HIP, mp.solutions.pose.PoseLandmark.RIGHT_KNEE),
            'left_knee': (mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.LEFT_KNEE, mp.solutions.pose.PoseLandmark.LEFT_ANKLE),
            'right_knee': (mp.solutions.pose.PoseLandmark.RIGHT_HIP, mp.solutions.pose.PoseLandmark.RIGHT_KNEE, mp.solutions.pose.PoseLandmark.RIGHT_ANKLE),
            'left_ankle': (mp.solutions.pose.PoseLandmark.LEFT_KNEE, mp.solutions.pose.PoseLandmark.LEFT_ANKLE, mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX),
            'right_ankle': (mp.solutions.pose.PoseLandmark.RIGHT_KNEE, mp.solutions.pose.PoseLandmark.RIGHT_ANKLE, mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX),
            'left_elbow': (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_ELBOW, mp.solutions.pose.PoseLandmark.LEFT_WRIST),
            'right_elbow': (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_ELBOW, mp.solutions.pose.PoseLandmark.RIGHT_WRIST),
            'left_shoulder': (mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_ELBOW),
            'right_shoulder': (mp.solutions.pose.PoseLandmark.RIGHT_HIP, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_ELBOW)
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
            mp.solutions.pose.POSE_CONNECTIONS
        )

    return image, angles

st.set_page_config(page_title="Movement AI", layout="wide")

lang = st.sidebar.selectbox("🌐 Language", ["English", "한국어", "中文"])
lang_code = {"English": "en", "한국어": "ko", "中文": "zh"}[lang]
t = TEXTS[lang_code]

st.title(t["title"])

uploaded_file = st.file_uploader(t["upload"], type=["jpg", "jpeg", "png"])
analyze = st.button(t["analyze"], type="primary")

@st.cache_resource
def load_pose():
    try:
        return mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None

if analyze and uploaded_file:
    pose = load_pose()

    if pose is None:
        st.stop()

    col1, col2 = st.columns(2)

    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.subheader(t["original"])
        st.image(image, use_column_width=True)

    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    processed, angles = process_image(img_cv, pose)

    with col2:
        st.subheader(t["result"])
        st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_column_width=True)

    if angles:
        st.subheader(t["angles"])

        df = pd.DataFrame(
            [(k.replace('_', ' ').title(), v) for k, v in angles.items() if v is not None],
            columns=["Joint", "Angle (deg)"]
        )

        name_map = {
            'left_hip': t["left_hip"], 'right_hip': t["right_hip"],
            'left_knee': t["left_knee"], 'right_knee': t["right_knee"],
            'left_ankle': t["left_ankle"], 'right_ankle': t["right_ankle"],
            'left_elbow': t["left_elbow"], 'right_elbow': t["right_elbow"],
            'left_shoulder': t["left_shoulder"], 'right_shoulder': t["right_shoulder"]
        }

        df["Joint"] = df["Joint"].str.lower().replace(name_map)

        st.dataframe(df)

        fig = go.Figure([
            go.Bar(x=df["Joint"], y=df["Angle (deg)"])
        ])

        fig.update_layout(
            title=t["chart"],
            xaxis_title="",
            yaxis_title="Degrees"
        )

        st.plotly_chart(fig)

    else:
        st.warning(t["no_pose"])

    pose.close()
