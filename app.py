import gradio as gr
import cv2
import mediapipe as mp
import numpy as np

# ================== MediaPipe 初始化 ==================
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    min_detection_confidence=0.5
)

# ================== 角度计算 ==================
def calc_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

# ================== 核心分析 ==================
def analyze(image):
    if image is None:
        return None, "No image"

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if not results.pose_landmarks:
        return image, "No pose detected"

    lm = results.pose_landmarks.landmark
    h, w, _ = image.shape

    def pt(i):
        return [lm[i].x * w, lm[i].y * h]

    mp_pose_enum = mp_pose.PoseLandmark

    joints = {
        "left_knee": (mp_pose_enum.LEFT_HIP, mp_pose_enum.LEFT_KNEE, mp_pose_enum.LEFT_ANKLE),
        "right_knee": (mp_pose_enum.RIGHT_HIP, mp_pose_enum.RIGHT_KNEE, mp_pose_enum.RIGHT_ANKLE),
        "left_elbow": (mp_pose_enum.LEFT_SHOULDER, mp_pose_enum.LEFT_ELBOW, mp_pose_enum.LEFT_WRIST),
        "right_elbow": (mp_pose_enum.RIGHT_SHOULDER, mp_pose_enum.RIGHT_ELBOW, mp_pose_enum.RIGHT_WRIST),
    }

    angles_text = []

    for name, (a, b, c) in joints.items():
        try:
            angle = calc_angle(pt(a), pt(b), pt(c))
            angles_text.append(f"{name}: {angle:.1f}°")
        except:
            pass

    mp_draw.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS
    )

    return image, "\n".join(angles_text)

# ================== Gradio UI ==================
demo = gr.Interface(
    fn=analyze,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Image(type="numpy", label="Result Image"),
        gr.Textbox(label="Joint Angles")
    ],
    title="AI Movement Analysis",
    description="Upload image → pose detection → joint angle analysis"
)

demo.launch(server_name="0.0.0.0", server_port=7860)
