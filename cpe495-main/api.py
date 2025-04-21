import cv2
import numpy as np
import os
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import mediapipe as mp
import json
import requests
from datetime import datetime

# โหลดโมเดล FaceNet และ MTCNN
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(image_size=160, margin=60, min_face_size=20)
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# โหลดฐานข้อมูลใบหน้าจากไฟล์ npy
face_database_path = r"C:\Users\Sujitra 582\Desktop\CPE495-Project\Face-Recognition-Attendance-System\face_database.npy"
if os.path.exists(face_database_path):
    data = np.load(face_database_path, allow_pickle=True).item()
    print(f"พบไฟล์ฐานข้อมูล: {face_database_path}")
    
    if isinstance(data, dict):
        face_embeddings = data
        print(f"โหลดฐานข้อมูลสำเร็จ! พบทั้งหมด {len(face_embeddings)} คน")
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
server_url = "http://172.20.10.2:5000/api/student-checks"

# โหลดวิดีโอ
video_path = r"C:\Users\Sujitra 582\Desktop\CPE495-Project\Face-Recognition-Attendance-System\video\SEC002.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("\n❌ Error: ไม่สามารถเปิดไฟล์วิดีโอได้!")
    exit()

# ดึงขนาดของเฟรมวิดีโอ
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_resized_width = 640  # ปรับขนาดเฟรมเพื่อเพิ่มความเร็ว
frame_resized_height = int((frame_resized_width / frame_width) * frame_height)

# โฟลเดอร์สำหรับบันทึกใบหน้าที่ตรวจพบ
output_folder = "detected_faces"
os.makedirs(output_folder, exist_ok=True)

frame_count = 0
saved_faces = set()

# อ่านเฟรมจากวิดีโอ
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ลดขนาดเฟรมเพื่อลดภาระการประมวลผล
    frame_resized = cv2.resize(frame, (frame_resized_width, frame_resized_height))

    boxes, _ = mtcnn.detect(frame_resized)
    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame_resized.shape[1], x2), min(frame_resized.shape[0], y2)
            face = frame_resized[y1:y2, x1:x2]

            if face.size == 0:
                continue

            if (x1, y1, x2, y2) not in saved_faces:
                face_filename = os.path.join(output_folder, f"face_{frame_count:04d}.jpg")
                cv2.imwrite(face_filename, face)
                saved_faces.add((x1, y1, x2, y2))
                frame_count += 1
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_resized, "Face Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Video - Face Detection", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 🔍 เปรียบเทียบภาพที่บันทึกกับฐานข้อมูล
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
            print(f"🔍 เปรียบเทียบ {face_file} กับ {db_name} | ค่าความคล้าย: {similarity:.2f}")  # Debug ค่าความคล้ายทุกคน
            if similarity > best_similarity:
                best_similarity = similarity
                name = db_name

        if best_similarity >= 0.75:  # ลด Threshold เป็น 0.75
            print(f"📌 ใบหน้าใน {face_file} ตรงกับ {name} (ค่าความคล้าย {best_similarity:.2f})")

            if name not in existing_names:
                # ส่งข้อมูลไปเซิร์ฟเวอร์
                attendance_data = {
                    "Student_ID": int(name),
                    "Course_ID": "CPE451",
                    "Check_Date": datetime.now().isoformat(),
                    "Check_Time": datetime.now().strftime("%H:%M:%S"),
                    "Check_Status": "Present"
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

# บันทึกผลลัพธ์ลงในไฟล์ JSON
json_output_path = "face_recognition_results.json"
with open(json_output_path, 'w', encoding='utf-8') as json_file:
    json.dump(attendance_records, json_file, ensure_ascii=False, indent=4)

print(f"✅ ผลลัพธ์ถูกบันทึกในไฟล์: {json_output_path}")
