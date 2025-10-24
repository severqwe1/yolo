import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# YOLO 모델 로드 (yolov8n.pt = 작은 버전)
model = YOLO("yolov8n.pt")

# RealSense 파이프라인 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 카메라 시작
profile = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # YOLOv8 객체 탐지
        results = model(color_image, verbose=False)

        annotated_frame = results[0].plot()  # YOLO 내장 시각화

        # depth 표시용
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.05),
            cv2.COLORMAP_JET
        )

        # 화면 병합
        combined = np.hstack((annotated_frame, depth_colormap))

        cv2.imshow("YOLOv8 + RealSense Depth", combined)

        # ESC 키 누르면 종료
        if cv2.waitKey(1) == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
