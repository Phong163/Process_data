# Hàm để thực hiện dự đoán với mô hình ONNX
import math
import cv2
import numpy as np
BODY_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Đầu và cổ
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Tay
    (5, 11), (6, 12),  # Thân trên
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)  # Chân
]
BODY_DELETE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
BODY_MATH3 = [
    (6,12), (5,11), (11,15), (12,16),(6,16),(5,15)
]

def make_landmark_timestep(points_frame):
    t1 = midpoint(points_frame[5],points_frame[6])
    t2 = midpoint(points_frame[11],points_frame[12])
    BODY_MATH1 = [
    (t1,t2,points_frame[14]), (t1,t2,points_frame[13]),(t2,points_frame[14],points_frame[16]),(t2,points_frame[13],points_frame[15])
    ]
    BODY_MATH2 = [
    (t1,t2), (t2,points_frame[14]), (t2,points_frame[13]), (points_frame[14],points_frame[16]),(points_frame[13],points_frame[15])
    ]
    c_lm = []
    
    x1,y1 = t1
    x2,y2 = t2
    for i, (x, y) in enumerate(points_frame):
        if x > 0 and y > 0 and i not in BODY_DELETE:
            c_lm.append(x)
            c_lm.append(y)
    c_lm.extend([x1, y1, x2, y2])
    for (a,b,c) in BODY_MATH1:
        angle1= calculate_angle(a,b,c)
        c_lm.append(angle1)
    for (a,b) in BODY_MATH2:
        angle2 = calculate_line_and_angle(a,b)
        avg_y = average_height(a,b)
        c_lm.append(angle2)
        c_lm.append(avg_y)
    return c_lm

def average_height(keypoint1, keypoint2):
    avg_y = (keypoint1[1] + keypoint2[1]) / 2
    return avg_y
def midpoint(point1, point2):
    x_mid = (point1[0] + point2[0]) / 2
    y_mid = (point1[1] + point2[1]) / 2
    return x_mid, y_mid

def draw_skeleton(image, points, t1, t2, thickness=1):
    h,w,_=image.shape
    points_frame = points * [w,h]
    cv2.circle(image, (int(t1[0]*w ), int(t1[1]*h)), 3, (255, 0, 0), -1)  # Vẽ trung điểm
    cv2.circle(image, (int(t2[0]*w), int(t2[1]*h)), 3, (255, 0, 0), -1)  # Vẽ trung điểm
    for i, (x, y) in enumerate(points_frame):
        if x > 0 and y > 0 and i not in BODY_DELETE:  # Kiểm tra xem điểm có hợp lệ không
            cv2.circle(image, (int(x), int(y)), 3, (255, 0, 0), -1)  # Vẽ điểm
    for (i, j) in BODY_CONNECTIONS:
        if points_frame[i][0] > 0 and points_frame[i][1] > 0 and points_frame[j][0] > 0 and points_frame[j][1] > 0:
            cv2.line(image, (int(points_frame[i][0]), int(points_frame[i][1])),
                     (int(points_frame[j][0]), int(points_frame[j][1])),
                     (255, 255, 255), thickness)
    cv2.line(image, (int(t1[0]*w), int(t1[1]*h )),
                 (int(t2[0]*w), int(t2[1]*h )),
                 (255, 255, 255), thickness)
    return image
def draw_skeleton_2(image, points, thickness=1):
    h, w, _ = image.shape
    points = np.array(points, dtype=np.float32)  # Chuyển points thành mảng NumPy
    points_frame = points * np.array([w, h])  # Nhân element-wise với [w, h]
    points_frame = points_frame.astype(int)  # Chuyển sang số nguyên để vẽ

    for i, (x, y) in enumerate(points_frame):
        if x > 0 and y > 0 and i not in BODY_DELETE:  # Kiểm tra xem điểm có hợp lệ không
            cv2.circle(image, (int(x), int(y)), 3, (255, 0, 0), -1)  # Vẽ điểm (màu xanh dương)

    for (i, j) in BODY_CONNECTIONS:
        if (points_frame[i][0] > 0 and points_frame[i][1] > 0 and 
            points_frame[j][0] > 0 and points_frame[j][1] > 0):
            cv2.line(image, (int(points_frame[i][0]), int(points_frame[i][1])),
                     (int(points_frame[j][0]), int(points_frame[j][1])),
                     (255, 255, 255), thickness)  # Vẽ đường nối (màu trắng)
    return image

def draw_fancy_rectangle(frame,size, a1, b1, a2, b2, color=(0, 255, 0), thickness=1, corner_length=15):
    w,h,z = frame.shape
    x1 = int((a1/size)*h)
    y1 = int((b1/size)*w)
    x2 = int((a2/size)*h)
    y2 = int((b2/size)*w)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return frame
def enlarge_bbox(frame, bbox,scale=1.2):
    x1, y1, x2, y2 = bbox
    frame_h, frame_w,_ = frame.shape
    
    w, h = x2 - x1, y2 - y1
    dw, dh = (w * scale - w) / 2, (h * scale - h) / 2

    new_x1, new_y1 = max(0, x1 - dw), max(0, y1 - dh)
    new_x2, new_y2 = min(frame_w, x2 + dw), min(frame_h, y2 + dh)

    return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]

def predict_with_onnx(session, input_tensor, input_name="images", output_name="output0"):
    # Resize ảnh về kích thước 640x640
    input_tensor = cv2.resize(input_tensor, (192, 192))

    # Kiểm tra số kênh ảnh
    if len(input_tensor.shape) == 2:
        input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_GRAY2RGB)
    elif input_tensor.shape[2] != 3:
        raise ValueError("Input frame must have 3 color channels (RGB).")

    # Chuẩn hóa ảnh: từ [0, 255] -> [0, 1] và đổi kiểu dữ liệu về float32
    input_tensor = input_tensor.astype(np.float32) / 255.0

    # Đổi shape từ (H, W, C) -> (1, C, H, W) để phù hợp với ONNX
    input_tensor = np.transpose(input_tensor, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    input_tensor = np.expand_dims(input_tensor, axis=0)  # Thêm batch_size = 1

    # Chạy mô hình ONNX
    outputs = session.run([output_name], {input_name: input_tensor})[0]
    return outputs

def draw_prediction_on_image(image, points):
    w ,h,_ = image.shape
    for person_keypoints in points:
            y, x = person_keypoints  # Đảm bảo lấy đúng thứ tự (y, x, score)
            cv2.circle(image, (int(y*h), int(x*w)), 1, (255, 0, 0), -1)

    return image

def draw_class_on_image(label, img, xy, color, scale):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = xy
    fontScale = scale
    fontColor = color
    thickness = 2
    lineType = 1
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img
def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))
def crop_img(frame, bbox):
    # Lấy tọa độ từ bbox
    x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    
    # Kiểm tra và giới hạn tọa độ không vượt quá kích thước ảnh
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(frame.shape[1], x_max)  # Chiều rộng
    y_max = min(frame.shape[0], y_max)  # Chiều cao

    scale = 1
    # Cắt ảnh
    cropped_image = frame[int(y_min *scale ):int(y_max * scale), int(x_min * scale):int(x_max * scale)]

    return cropped_image

import numpy as np

def point_in_frame(frame, inp, bbox, points):
    h, w,_ = frame.shape
    y, x, _ = inp.shape  # Kích thước box sau resize (192,192)
    x_min, y_min, x_max, y_max = bbox  
    # Scale từ box resize về bbox gốc trên frame
    points_inp = points  * [y, x]

    # Chuyển từ bbox về frame gốc
    points_frame = np.zeros_like(points)
    points_frame[:, 0] = x_min + points_inp[:, 1]  # X
    points_frame[:, 1] = y_min + points_inp[:, 0]  # Y  

    points_frame = points_frame / [w,h]
    t1 = midpoint(points_frame[5],points_frame[6])
    t2 = midpoint(points_frame[11],points_frame[12])

    return points_frame, t1, t2

    


def calculate_line_to_height_ratio(A, B, frame_distance):
    # Tính độ dài của đoạn đường thẳng giữa hai điểm (distance = sqrt((x2 - x1)^2 + (y2 - y1)^2))
    distance = np.linalg.norm(np.array(A) - np.array(B))
    
    # Tính tỷ lệ giữa chiều dài đoạn đường thẳng và chiều cao của frame
    ratio = distance / frame_distance
    
    return ratio
def box_rescale(frame, size, x1, y1, x2, y2):
    h, w, i = frame.shape
    x1 = int((x1/size)*w)
    y1 = int((y1/size)*h)
    x2 = int((x2/size)*w)
    y2 = int((y2/size)*h)
    return x1, y1, x2, y2
def kp_rescale(frame, size, x, y):
    h, w, i = frame.shape
    kp_x = int((x/size)*w)
    kp_y = int((y/size)*h)
    return kp_x, kp_y
# Hàm tính góc giữa 3 điểm toạ độ
def calculate_angle(A, C, B):
    # Vector AB
    AB = np.array([B[0] - A[0], B[1] - A[1]])
    # Vector BC
    BC = np.array([C[0] - B[0], C[1] - B[1]])

    # Tích vô hướng của hai vector
    dot_product = np.dot(AB, BC)
    # Độ dài của vector
    norm_AB = np.linalg.norm(AB)
    norm_BC = np.linalg.norm(BC)

    # Tính góc bằng arccos
    angle_rad = np.arccos(dot_product / (norm_AB * norm_BC))
    # Chuyển đổi sang độ
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def calculate_velocity(pose_data):
    num_frames = pose_data.shape[0]
    velocity = np.zeros_like(pose_data)  # Vận tốc cùng kích thước với tọa độ pose
    
    for i in range(1, num_frames):
        velocity[i] = pose_data[i] - pose_data[i - 1]
    
    return velocity

# Hàm tính gia tốc (acceleration) từ vận tốc
def calculate_acceleration(velocity):
    num_frames = velocity.shape[0]
    acceleration = np.zeros_like(velocity)  # Gia tốc cùng kích thước với vận tốc
    
    for i in range(1, num_frames):
        acceleration[i] = velocity[i] - velocity[i - 1]
    
    return acceleration

def calculate_line_and_angle(A, B):
    # Tính góc giữa đường thẳng và bề ngang
    dx = B[0] - A[0]
    dy = B[1] - A[1]
    angle_rad = math.atan2(dy, dx)  # Góc theo radian
    angle_deg = math.degrees(angle_rad)  # Chuyển sang độ

    # Đảm bảo góc nằm trong khoảng [0, 180] độ
    if angle_deg < 0:
        angle_deg += 180
    return angle_deg
def in_img(frame):
    img = frame[:, :, ::-1].transpose(2, 0, 1)  # Chuyển từ BGR sang RGB và đổi trục
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0  # Chuẩn hóa về [0,1]
    img = np.expand_dims(img, axis=0)  # Thêm batch dimension
    return img
