import time
import os
import cv2
import numpy as np
import onnxruntime as ort
import torch
from tracker.tracker import BYTETracker
from utils import *
from general import non_max_suppression_2
from config import *

# Thiết lập thiết bị (GPU hoặc CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLO model
yolo_weights = "weights/yolov8l-pose.onnx"
yolo_session = ort.InferenceSession(yolo_weights)
input_name = yolo_session.get_inputs()[0].name
output_name = yolo_session.get_outputs()[0].name

def detect_yolo(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    frame = frame.astype(np.float32) / 255.0  # Normalize to [0, 1]
    input_tensor = np.expand_dims(np.transpose(frame, (2, 0, 1)), axis=0).astype(np.float32)
    detected = yolo_session.run(None, {input_name: input_tensor})[0]
    detected = non_max_suppression_2(detected, conf_thres, iou_thres, max_det=max_det)
    return detected

def rescale(frame, size, x1, y1, x2, y2):
    h, w, _ = frame.shape
    return (int((x1 / size) * w), int((y1 / size) * h), 
            int((x2 / size) * w), int((y2 / size) * h))

def kp_rescale(frame, size, kp_x, kp_y):
    h, w, _ = frame.shape
    return int((kp_x / size) * w), int((kp_y / size) * h)

def convert_visibility(vis):
    if 0 <= vis <= 0.2:
        return 0
    elif 0.2 < vis <= 0.55:
        return 1
    return 2

def xyxy2xywh(image_width, image_height, x_min, y_min, x_max, y_max):
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    return x_center, y_center, width, height

def save_frame(frame, output_frame_folder, video_name, frame_counter, global_frame_counter):
    frame_filename = os.path.join(output_frame_folder, f"{video_name}_{global_frame_counter[0]:04d}.jpg")
    cv2.imwrite(frame_filename, frame)
    return frame_filename

def save_labels(labels, output_label_folder, frame_filename):
    label_filename = os.path.join(output_label_folder, 
                                 os.path.splitext(os.path.basename(frame_filename))[0] + ".txt")
    with open(label_filename, 'w') as f:
        for label in labels:
            modified_label = [0] + label  # Add class_id = 0
            f.write(' '.join(map(str, modified_label)) + '\n')

def process_video(video_path, output_frame_folder, output_label_folder, global_frame_counter):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_frame_folder, exist_ok=True)
    os.makedirs(output_label_folder, exist_ok=True)

    tracker = BYTETracker(track_thresh=0.5, match_thresh=0.8, track_buffer=30, mot20=False)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FPS, 15)

    frame_counter = 0
    prev_time = time.time()
    track_labels = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        frame2 = frame.copy()
        frame_counter += 1
        global_frame_counter[0] += 1  # Increment global frame counter
        if frame_counter % 10 ==0:
            # Resize frame for processing
            size = 640
            frame1 = cv2.resize(frame, (size, size))
            detected = detect_yolo(frame1)
            labels = []
            if np.any(detected):
                detections = []
                for detect in detected[0]:
                    box_score = detect[0:5]  # [x1, y1, x2, y2, conf]
                    keypoints = detect[5:]   # [kp1_x, kp1_y, kp1_vis, ...]
                    x1, y1, x2, y2 = rescale(frame, size, *box_score[:4])
                    keypoints_new = []
                    for i in range(0, len(keypoints), 3):
                        kp_x, kp_y, vis = keypoints[i:i+3]
                        x, y = kp_rescale(frame, size, kp_x, kp_y)
                        vis_converted = convert_visibility(vis)
                        keypoints_new.extend([x, y, float(vis_converted)])
                    detections.append([x1, y1, x2, y2, box_score[4]] + keypoints_new)

                tracks = tracker.update(np.array(detections))
            else:
                tracks = []

            for track in tracks:
                track_id = track.track_id
                x1, y1, x2, y2 = map(float, track.tlbr.tolist()[:4])
                keypoints_new = track.keypoints.tolist() if track.keypoints is not None else [0] * 51
                x_center, y_center, width, height = xyxy2xywh(w, h, x1, y1, x2, y2)

                # Normalize coordinates
                x1_norm = "{:.6f}".format(x_center / w)
                y1_norm = "{:.6f}".format(y_center / h)
                w_norm = "{:.6f}".format(width / w)
                h_norm = "{:.6f}".format(height / h)

                keypoints_norm = []
                lstm_input = []
                for i in range(0, len(keypoints_new), 3):
                    kp_x, kp_y, vis = keypoints_new[i:i+3]
                    kp_x_norm = "{:.6f}".format(kp_x / w if kp_x > 0 else 0)
                    kp_y_norm = "{:.6f}".format(kp_y / h if kp_y > 0 else 0)
                    vis_converted = convert_visibility(vis)
                    lstm_input.append([float(kp_x_norm), float(kp_y_norm)])
                    keypoints_norm.extend([kp_x_norm, kp_y_norm, "{:.6f}".format(vis_converted)])

                frame2 = draw_skeleton_2(frame2, lstm_input)
                labels.append([x1_norm, y1_norm, w_norm, h_norm] + keypoints_norm)

                # Draw bounding box
                if track_id not in track_labels:
                    track_labels[track_id] = {'label': "Unknown", 'clr': (255, 255, 255)}
                frame2 = cv2.rectangle(frame2, (int(x1), int(y1)), (int(x2), int(y2)), 
                                    track_labels[track_id]['clr'], 2)
                frame2 = cv2.putText(frame2, f"{track_labels[track_id]['label']} id:{track_id}",
                                    (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 
                                    1, track_labels[track_id]['clr'], 2)

            # Save frame and labels
            frame_filename = save_frame(frame, output_frame_folder, video_name, frame_counter, global_frame_counter)
            save_labels(labels, output_label_folder, frame_filename)

        # Calculate and display FPS
        # current_time = time.time()
        # fps = 1 / (current_time - prev_time)
        # prev_time = current_time
        frame2 = draw_class_on_image(f"Frame:{global_frame_counter[0]}", frame2, 
                                   (10, 30), (0, 255, 0), 1)

        cv2.imshow('Pose Estimation', frame2)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    print(f"✅ Processed {video_name} with {frame_counter} frames")

if __name__ == "__main__":
    video_dir = r"C:\Users\OS\Desktop\ActionProject\Home_01\Videos"
    output_base_dir = r"C:\Users\OS\Desktop\ActionProject\datasets"
    output_frame_folder = os.path.join(output_base_dir, "images")
    output_label_folder = os.path.join(output_base_dir, "labels")

    # Supported video extensions
    video_extensions = ('.avi', '.mp4', '.mov', '.mkv')

    # Global frame counter to ensure unique filenames across all videos
    global_frame_counter = [0]  # Using a list to allow modification in nested functions

    # Iterate through all videos in the directory
    for video_file in os.listdir(video_dir):
        if video_file.lower().endswith(video_extensions):
            video_path = os.path.join(video_dir, video_file)
            process_video(video_path, output_frame_folder, output_label_folder, global_frame_counter)

    cv2.destroyAllWindows()
    print(f"✅ All videos processed. Results saved to {output_frame_folder} and {output_label_folder}")