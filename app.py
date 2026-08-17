
import streamlit as st
import cv2
import numpy as np
import tempfile
import mediapipe as mp

from scipy.signal import find_peaks
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


st.set_page_config(
    page_title="縄跳び解析ツール",
    page_icon="🏃"
)

st.title("縄跳び解析ツール")
st.write("縄跳びの動画を選択すると、回数・時間・ペースを解析します。")


# ==========================================
# 動画選択
# ==========================================

uploaded_video = st.file_uploader(
    "解析する動画を選択",
    type=["mp4", "mov"]
)


if uploaded_video is not None:

    st.success("動画を受け取りました！")

    if st.button("解析する"):

        with st.spinner("解析中..."):

            # --------------------------------
            # アップロード動画を一時保存
            # --------------------------------

            suffix = "." + uploaded_video.name.split(".")[-1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as tmp:

                tmp.write(uploaded_video.getbuffer())
                video_path = tmp.name


            # --------------------------------
            # MediaPipe準備
            # --------------------------------

            base_options = python.BaseOptions(
                model_asset_path="pose_landmarker_lite.task"
            )

            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO
            )

            landmarker = (
                vision.PoseLandmarker
                .create_from_options(options)
            )


            # --------------------------------
            # 動画解析
            # --------------------------------

            video = cv2.VideoCapture(video_path)

            fps = video.get(cv2.CAP_PROP_FPS)
            frame_count = int(
                video.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            full_hip_y = []
            full_times = []

            frame_number = 0

            while frame_number < frame_count:

                success, frame = video.read()

                if not success:
                    break

                # 2フレームに1回解析
                if frame_number % 2 == 0:

                    rgb_frame = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb_frame
                    )

                    timestamp_ms = int(
                        frame_number * 1000 / fps
                    )

                    result = landmarker.detect_for_video(
                        mp_image,
                        timestamp_ms
                    )

                    if result.pose_landmarks:

                        landmarks = result.pose_landmarks[0]

                        left_hip = landmarks[23]
                        right_hip = landmarks[24]

                        hip_y = (
                            left_hip.y +
                            right_hip.y
                        ) / 2

                        full_hip_y.append(hip_y)
                        full_times.append(
                            frame_number / fps
                        )

                frame_number += 1


            video.release()
            landmarker.close()


            # --------------------------------
            # ジャンプ解析
            # --------------------------------

            if len(full_hip_y) >= 3:

                hip_array = np.array(full_hip_y)

                smooth_hip_y = np.convolve(
                    hip_array,
                    np.ones(3) / 3,
                    mode="same"
                )

                jump_points, _ = find_peaks(
                    -smooth_hip_y,
                    distance=4,
                    prominence=0.036
                )

                jump_count = len(jump_points)

                jump_times = np.array(
                    full_times
                )[jump_points]


                # ----------------------------
                # 実際に跳んでいた時間
                # ----------------------------

                jumping_time = 0

                for i in range(len(jump_times) - 1):

                    gap = (
                        jump_times[i + 1]
                        - jump_times[i]
                    )

                    if gap <= 1.5:
                        jumping_time += gap


                # ----------------------------
                # 結果
                # ----------------------------

                if jumping_time > 0:

                    pace = (
                        jump_count /
                        jumping_time *
                        60
                    )

                    st.success("解析完了！")

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "ジャンプ回数",
                        f"{jump_count} 回"
                    )

                    col2.metric(
                        "跳んでいた時間",
                        f"{jumping_time:.2f} 秒"
                    )

                    col3.metric(
                        "ペース",
                        f"{pace:.1f} 回/分"
                    )

                else:

                    st.error(
                        "縄跳びを十分に検出できませんでした。"
                    )

            else:

                st.error(
                    "動画から姿勢を十分に検出できませんでした。"
                )
