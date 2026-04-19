import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tempfile
import os
from PIL import Image
import plotly.graph_objects as go

# ================== 多语言 ==================
TEXTS = {
    "en": {
        "title": "Movement Assessment & AI Platform",
        "upload_image": "Upload Image (JPG/PNG)",
        "upload_video": "Upload Video (MP4/AVI/MOV)",
        "analyze_btn": "Analyze",
        "raw_image": "Original Image",
        "result_image": "Analysis Result",
        "joint_angles": "Joint Angles",
        "angle_chart": "Angle Chart",
        "video_analysis": "Video Analysis",
        "angle_trend": "Joint Angle Trends over Time",
        "no_pose": "No human pose detected.",
        "processing": "Processing, please wait...",
        "left_knee": "Left Knee", "right_knee": "Right Knee",
        "left_hip": "Left Hip", "right_hip": "Right Hip",
        "left_ankle": "Left Ankle", "right_ankle": "Right Ankle",
        "left_elbow": "Left Elbow", "right_elbow": "Right Elbow",
        "left_shoulder": "Left Shoulder", "right_shoulder": "Right Shoulder",
        "download_data": "Download Angle Data (CSV)"
    },
    "ko": {
        "title": "움직임 평가 및 AI 플랫폼",
        "upload_image": "이미지 업로드 (JPG/PNG)",
        "upload_video": "비디오 업로드 (MP4/AVI/MOV)",
        "analyze_btn": "분석",
        "raw_image": "원본 이미지",
        "result_image": "분석 결과",
        "joint_angles": "관절 각도",
        "angle_chart": "각도 차트",
        "video_analysis": "비디오 분석",
        "angle_trend": "각도 변화 추이",
        "no_pose": "인체 자세가 감지되지 않았습니다.",
        "processing": "처리 중입니다. 잠시만 기다려주세요...",
        "left_knee": "왼쪽 무릎", "right_knee": "오른쪽 무릎",
        "left_hip": "왼쪽 엉덩이", "right_hip": "오른쪽 엉덩이",
        "left_ankle": "왼쪽 발목", "right_ankle": "오른쪽 발목",
        "left_elbow": "왼쪽 팔꿈치", "right_elbow": "오른쪽 팔꿈치",
        "left_shoulder": "왼쪽 어깨", "right_shoulder": "오른쪽 어깨",
        "download_data": "각도 데이터 다운로드 (CSV)"
    },
    "zh": {
        "title": "运动评估与AI平台",
        "upload_image": "上传图片 (JPG/PNG)",
        "upload_video": "上传视频 (MP4/AVI/MOV)",
        "analyze_btn": "分析",
        "raw_image": "原始图片",
        "result_image": "分析结果",
        "joint_angles": "关节角度",
        "angle_chart": "角度图表",
        "video_analysis": "视频分析",
        "angle_trend": "角度变化趋势",
        "no_pose": "未检测到人体姿态。",
        "processing": "处理中，请稍候...",
        "left_knee": "左膝", "right_knee": "右膝",
        "left_hip": "左髋", "right_hip": "右髋",
        "left_ankle": "左踝", "right_ankle": "右踝",
        "left_elbow": "左肘", "right_elbow": "右肘",
        "left_shoulder": "左肩", "right_shoulder": "右肩",
        "download_data": "下载角度数据 (CSV)"
    }
}

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def process_frame(image, pose):
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
        for name, (i,j,k) in joints.items():
            try:
                a = [landmarks[i].x*w, landmarks[i].y*h]
                b = [landmarks[j].x*w, landmarks[j].y*h]
                c = [landmarks[k].x*w, landmarks[k].y*h]
                angles[name] = calculate_angle(a,b,c)
            except:
                angles[name] = None
        mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
    return image, angles

st.set_page_config(page_title="Movement AI", layout="wide")

lang = st.sidebar.selectbox("🌐 Language", ["English", "한국어", "中文"])
lang_code = {"English": "en", "한국어": "ko", "中文": "zh"}[lang]
t = TEXTS[lang_code]

st.title(t["title"])

with st.sidebar:
    media_type = st.radio("", ["📸 Image", "🎥 Video"])
    uploaded_file = None
    if media_type == "📸 Image":
        uploaded_file = st.file_uploader(t["upload_image"], type=["jpg","jpeg","png"])
    else:
        uploaded_file = st.file_uploader(t["upload_video"], type=["mp4","avi","mov"])
    analyze = st.button(t["analyze_btn"], type="primary", use_container_width=True)

@st.cache_resource
def load_pose():
    return mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

if analyze and uploaded_file:
    pose = load_pose()
    
    if media_type == "📸 Image":
        col1, col2 = st.columns(2)
        image = Image.open(uploaded_file)
        with col1:
            st.subheader(t["raw_image"])
            st.image(image, use_container_width=True)
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        processed, angles = process_frame(img_cv, pose)
        with col2:
            st.subheader(t["result_image"])
            st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_container_width=True)
        if angles:
            st.subheader(t["joint_angles"])
            df = pd.DataFrame([(k.replace('_',' ').title(), v) for k,v in angles.items() if v], columns=["Joint", "Angle (deg)"])
            name_map = {
                'left_hip': t["left_hip"], 'right_hip': t["right_hip"],
                'left_knee': t["left_knee"], 'right_knee': t["right_knee"],
                'left_ankle': t["left_ankle"], 'right_ankle': t["right_ankle"],
                'left_elbow': t["left_elbow"], 'right_elbow': t["right_elbow"],
                'left_shoulder': t["left_shoulder"], 'right_shoulder': t["right_shoulder"]
            }
            df["Joint"] = df["Joint"].str.lower().replace(name_map)
            st.dataframe(df, use_container_width=True)
            fig = go.Figure([go.Bar(x=df["Joint"], y=df["Angle (deg)"], marker_color='lightcoral')])
            fig.update_layout(title=t["angle_chart"], xaxis_title="", yaxis_title="Degrees", template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(t["no_pose"])
    
    else:  # 视频分析
        st.subheader(t["video_analysis"])
        with st.spinner(t["processing"]):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
            frame_angles = []
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                _, angles = process_frame(frame, pose)
                if angles:
                    angles['frame'] = frame_idx
                    frame_angles.append(angles)
                frame_idx += 1
            cap.release()
            os.unlink(tfile.name)
        
        if frame_angles:
            df_angles = pd.DataFrame(frame_angles)
            plot_joints = ['left_knee', 'right_knee', 'left_hip', 'right_hip']
            existing = [j for j in plot_joints if j in df_angles.columns]
            if existing:
                name_map = {'left_knee': t["left_knee"], 'right_knee': t["right_knee'], 'left_hip': t["left_hip"], 'right_hip': t["right_hip"]}
                fig = go.Figure()
                for j in existing:
                    fig.add_trace(go.Scatter(x=df_angles['frame'], y=df_angles[j], mode='lines', name=name_map[j]))
                fig.update_layout(title=t["angle_trend"], xaxis_title="Frame", yaxis_title="Angle (deg)", template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
            csv_data = df_angles.to_csv(index=False).encode('utf-8')
            st.download_button(t["download_data"], csv_data, "joint_angles.csv", "text/csv")
        else:
            st.warning(t["no_pose"])
    
    pose.close()
elif analyze and not uploaded_file:
    st.warning("Please upload a file first.")
