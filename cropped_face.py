import os
import cv2
import torch
import numpy as np
from retinaface import RetinaFace  # 确保安装了 RetinaFace 的 Python 包
from PIL import Image
from torchvision import transforms
import face_alignment  # 用于面部对齐

# 设置输出文件夹路径
output_root = "/data/lyh/Affwild2/cropped_face"
train_output_dir = os.path.join(output_root, "train")
val_output_dir = os.path.join(output_root, "val")

# 创建输出文件夹
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(val_output_dir, exist_ok=True)

# 数据转换：将检测到的人脸裁剪并归一化为 224x224
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 初始化面部对齐器
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda:1' if torch.cuda.is_available() else 'cpu')

def process_video(video_path, output_dir):
    # 获取视频文件名，不含扩展名
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    previous_face = None

    print(f"Processing video: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Finished processing video: {video_path}")
            break

        frame_count += 1
        print(f"Processing frame {frame_count} of video {video_name}")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 检测人脸
        faces = RetinaFace.detect_faces(frame_rgb)
        if isinstance(faces, dict) and len(faces) > 0:
            print(f"Detected {len(faces)} face(s) in frame {frame_count}")
            # 假设只使用检测到的第一个人脸
            face = faces[list(faces.keys())[0]]
            facial_area = face['facial_area']
            x1, y1, x2, y2 = facial_area
            cropped_face = frame_rgb[y1:y2, x1:x2]

            # 对齐人脸
            landmarks = fa.get_landmarks(frame_rgb, [facial_area])
            if landmarks is not None:
                print(f"Aligning face in frame {frame_count}")
                # 假设只使用第一个检测到的人脸的关键点
                aligned_face = fa.align(frame_rgb, landmarks[0], facial_area, align=True)
                cropped_face = aligned_face
            else:
                print(f"No landmarks found for face in frame {frame_count}")

            previous_face = cropped_face
        else:
            print(f"No face detected in frame {frame_count}, using previous face")
            # 如果未检测到人脸，使用上一帧的人脸
            if previous_face is not None:
                cropped_face = previous_face
            else:
                # 如果之前没有人脸可用，则跳过该帧
                print(f"No previous face available for frame {frame_count}, skipping frame")
                continue

        # 将裁剪到的人脸转为 PIL 图像并进行归一化处理
        cropped_face_resized = cv2.resize(cropped_face, (224, 224))
        face_pil = Image.fromarray(cropped_face_resized)
        face_tensor = transform(face_pil)

        # 将人脸保存到输出文件夹
        output_image_path = os.path.join(video_output_dir, f"{frame_count:05d}.jpg")
        save_image = transforms.ToPILImage()(face_tensor)
        save_image.save(output_image_path)
        print(f"Saved frame {frame_count} to {output_image_path}")

    cap.release()
    print(f"Released video capture for {video_path}")

# 处理所有视频文件
def process_all_videos(video_dir, output_dir):
    print(f"Starting to process videos in directory: {video_dir}")
    for root, _, files in os.walk(video_dir):
        for file in files:
            if file.endswith(('.mp4', '.avi')):
                video_path = os.path.join(root, file)
                print(f"Found video file: {file}")
                process_video(video_path, output_dir)
    print(f"Finished processing all videos in directory: {video_dir}")

# 处理 train 和 val 数据集
train_video_dir = "/data/lyh/Affwild2/raw_videos/train"  # 替换为你的 train 视频路径
val_video_dir = "/data/lyh/Affwild2/raw_videos/val"      # 替换为你的 val 视频路径

process_all_videos(train_video_dir, train_output_dir)
process_all_videos(val_video_dir, val_output_dir)

print("视频处理完成。")
