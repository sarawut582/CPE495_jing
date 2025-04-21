import cv2
import numpy as np
import os
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import mediapipe as mp
import json
import requests
from datetime import datetime, timedelta, timezone
import time

# โหลดโมเดล FaceNet และ MTCNN
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(image_size=160, margin=60, min_face_size=20)
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# โหลดฐานข้อมูลใบหน้าจากไฟล์ npy
face_database_path = "face_database.npy"
if os.path.exists(face_database_path):
    data = np.load(face_database_path, allow_pickle=True).item()
    if isinstance(data, dict):
        face_embeddings = data
    else:
        print("❌ ข้อมูลใน face_database.npy ไม่ใช่ dictionary!")
        exit()
else:
    print("❌ ไม่พบไฟล์ face_database.npy!")
    exit()

# ฟังก์ชันสร้าง face embedding
def recognize_face_with_mtcnn(face_resized):
    try:
        if face_resized is None or face_resized.size == 0:
            return None
        face_resized = cv2.resize(face_resized, (160, 160))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_tensor = torch.tensor(face_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        with torch.no_grad():
            embedding = facenet_model(face_tensor)
        return embedding.squeeze().numpy()
    except Exception as e:
        print("⚠️ Error in face embedding:", e)
        return None

# ฟังก์ชันคำนวณความคล้ายคลึง (cosine similarity)
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# URL ของเซิร์ฟเวอร์ API
server_url = "https://a443-49-237-33-240.ngrok-free.app/api/student-checks"

# โฟลเดอร์วิดีโอ
video_folder = "recorded_videos"

# ฟังก์ชันดึงไฟล์วิดีโอที่มี .done
def get_latest_videos():
    videos = []
    for file in os.listdir(video_folder):
        if file.endswith(".mp4") and os.path.exists(os.path.join(video_folder, file + ".done")):
            videos.append(os.path.join(video_folder, file))
    return sorted(videos, key=os.path.getctime, reverse=True)

# Timezone ประเทศไทย
THAI_TIMEZONE = timezone(timedelta(hours=7))

while True:
    latest_videos = get_latest_videos()
    if not latest_videos:
        print("❌ ไม่พบวิดีโอในโฟลเดอร์ recorded_videos! กำลังรอไฟล์ใหม่...")
        time.sleep(10)
        continue

    for video_path in latest_videos:
        print(f"📂 กำลังประมวลผลวิดีโอ: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ ไม่สามารถเปิดไฟล์วิดีโอได้!")
            continue

        output_folder = "detected_faces"
        os.makedirs(output_folder, exist_ok=True)

        frame_count = 0
        saved_faces = set()

        frame_index = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while frame_index < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                break

            boxes, _ = mtcnn.detect(frame)
            if boxes is not None:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    face = frame[y1:y2, x1:x2]

                    if face.size == 0:
                        continue

                    if (x1, y1, x2, y2) not in saved_faces:
                        face_filename = os.path.join(output_folder, f"face_{frame_count:04d}.jpg")
                        cv2.imwrite(face_filename, face)
                        saved_faces.add((x1, y1, x2, y2))
                        frame_count += 1

            frame_index += 10

        cap.release()
        cv2.destroyAllWindows()

        print("✅ การตรวจจับใบหน้าเสร็จสิ้น!")
        print("🔍 กำลังเปรียบเทียบใบหน้ากับฐานข้อมูล...")

        existing_names = set()
        attendance_records = []

        for face_file in os.listdir(output_folder):
            face_path = os.path.join(output_folder, face_file)
            face_img = cv2.imread(face_path)
            face_embedding = recognize_face_with_mtcnn(face_img)

            if face_embedding is not None:
                name = "Unknown"
                best_similarity = float("-inf")

                for db_name, db_embedding in face_embeddings.items():
                    similarity = cosine_similarity(face_embedding, db_embedding)
                    print(f"🔍 เปรียบเทียบ {face_file} กับ {db_name} | ค่าความคล้าย: {similarity:.2f}")
                    if similarity > best_similarity:
                        best_similarity = similarity
                        name = db_name

                if best_similarity >= 0.75:
                    print(f"📌 ใบหน้าใน {face_file} ตรงกับ {name} (ค่าความคล้าย {best_similarity:.2f})")

                    if name not in existing_names:
                        now_th = datetime.now(THAI_TIMEZONE)

                        attendance_data = {
                            "Student_ID": int(name),
                            "Course_ID": "CPE451",
                            "Check_Date": now_th.astimezone(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z'),
                            "Check_Time": now_th.strftime("%H:%M:%S")
                        }

                        try:
                            headers = {"Content-Type": "application/json"}
                            response = requests.post(server_url, json=attendance_data, headers=headers)

                            if response.status_code == 200:
                                print(f"✅ ส่งข้อมูลสำเร็จ: {name}")
                            else:
                                print(f"❌ ส่งข้อมูลล้มเหลว: {name} | Status Code: {response.status_code} | Response: {response.text}")
                        except Exception as e:
                            print(f"⚠️ Error sending data: {e}")

                        existing_names.add(name)

            os.remove(face_path)
            print(f"❌ ลบไฟล์: {face_file}")

        print("✅ การเปรียบเทียบเสร็จสิ้น!")

        done_file = video_path + ".done"
        if os.path.exists(done_file):
            os.remove(done_file)
            print(f"✅ ลบไฟล์ .done สำหรับ {video_path}")

    print("🔄 รอวิดีโอใหม่...")
    time.sleep(10)
