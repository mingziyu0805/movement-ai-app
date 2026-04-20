import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go

# ================== 多语言 ==================
TEXTS = {
    "en": {"title": "Movement AI", "upload": "Upload Image", "analyze": "Analyze",
           "original": "Original Image", "result": "Result",
           "angles": "Angles", "chart": "Chart", "no_pose": "No pose detected"},

    "ko": {"title": "AI 움직임 분석", "upload": "이미지 업로드", "analyze": "분석",
           "original": "원본", "result": "결과",
           "angles": "관절", "chart": "그래프", "no_pose": "인체 없음"},

    "zh": {"title": "运动AI分析", "upload": "上传图片", "analyze": "分析",
           "original": "原图", "result": "结果",
           "angles": "关节", "chart": "图表", "no_pose": "未检测到人体"}
}

# ================== 🔥 关键修复：安全加载 MediaPipe ==================
def load_pose():
    try:
        import mediapipe as mp

        # 🔥 防止 solutions 丢失（你现在问题核心）
        if not hasattr(mp, "solutions"):
            raise ImportError("mediapipe 安装不完整（missing solutions）")

        pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5
        )

        return pose, mp

    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None, None


# ================== 角度计算 ==================
def calc_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle


# ================== pose处理 ==================
def process(img, pose, mp):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    angles = {}

    if not results or not results.pose_landmarks:
        return img, angles

    lm = results.pose_landmarks.landmark
    h, w, _ = img.shape

    def pt(i):
        return [lm[i].x * w, lm[i].y * h]

    mp_pose = mp.solutions.pose.PoseLandmark

    joints = {
        "left_knee": (mp_pose.LEFT_HIP, mp_pose.LEFT_KNEE, mp_pose.LEFT_ANKLE),
        "right_knee": (mp_pose.RIGHT_HIP, mp_pose.RIGHT_KNEE, mp_pose.RIGHT_ANKLE),
        "left_elbow": (mp_pose.LEFT_SHOULDER, mp_pose.LEFT_ELBOW, mp_pose.LEFT_WRIST),
        "right_elbow": (mp_pose.RIGHT_SHOULDER, mp_pose.RIGHT_ELBOW, mp_pose.RIGHT_WRIST),
        "left_shoulder": (mp_pose.LEFT_HIP, mp_pose.LEFT_SHOULDER, mp_pose.LEFT_ELBOW),
        "right_shoulder": (mp_pose.RIGHT_HIP, mp_pose.RIGHT_SHOULDER, mp_pose.RIGHT_ELBOW),
    }

    for k, (a, b, c) in joints.items():
        try:
            angles[k] = calc_angle(pt(a), pt(b), pt(c))
        except:
            angles[k] = None

    mp.solutions.drawing_utils.draw_landmarks(
        img,
        results.pose_landmarks,
        mp.solutions.pose.POSE_CONNECTIONS
    )

    return img, angles


# ================== UI ==================
st.set_page_config(page_title="AI Movement", layout="wide")

lang = st.selectbox("Language", ["English", "한국어", "中文"])
code = {"English": "en", "한국어": "ko", "中文": "zh"}[lang]
t = TEXTS[code]

st.title(t["title"])

file = st.file_uploader(t["upload"], type=["jpg", "png", "jpeg"])
btn = st.button(t["analyze"])

pose, mp = load_pose()

if btn and file and pose:

    img = Image.open(file).convert("RGB")
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(t["original"])
        st.image(img)

    out, angles = process(img, pose, mp)

    with col2:
        st.subheader(t["result"])
        st.image(out)

    if angles:
        df = pd.DataFrame(
            [(k, v) for k, v in angles.items() if v is not None],
            columns=["Joint", "Angle"]
        )

        st.dataframe(df)

        fig = go.Figure([go.Bar(x=df["Joint"], y=df["Angle"])])
        st.plotly_chart(fig)

    else:
        st.warning(t["no_pose"])

    pose.close()
