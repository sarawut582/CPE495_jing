import cv2
import numpy as np
import torch
import os
from facenet_pytorch import MTCNN, InceptionResnetV1

# โหลดโมเดล MTCNN สำหรับตรวจจับใบหน้า และ FaceNet สำหรับสร้าง Embedding
mtcnn = MTCNN(image_size=160, margin=40, min_face_size=20)
facenet = InceptionResnetV1(pretrained='vggface2').eval()

# โฟลเดอร์ที่เก็บรูปภาพใบหน้า
dataset_path = r"C:\Users\Sujitra 582\Desktop\CPE495-Project\Face-Recognition-Attendance-System\faces_dataset"
if not os.path.exists(dataset_path):
    print(f"❌ ไม่พบโฟลเดอร์ '{dataset_path}' โปรดตรวจสอบอีกครั้ง")
    exit() 
    

face_embeddings = {}

# อ่านโฟลเดอร์บุคคลจากฐานข้อมูล
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    
    if os.path.isdir(person_folder):  # ตรวจสอบว่าเป็นโฟลเดอร์
        embeddings_list = []
        for filename in os.listdir(person_folder):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(person_folder, filename)

                # โหลดภาพ
                img = cv2.imread(image_path)
                if img is None:
                    print(f"⚠️ อ่านไฟล์ไม่สำเร็จ: {image_path}")
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # ตรวจจับใบหน้าและ Crop
                face = mtcnn(img_rgb)
                
                if face is not None:
                    img_tensor = face.unsqueeze(0)  # เพิ่ม Batch Dimension
                    
                    # ดึง Face Embedding
                    with torch.no_grad():
                        embedding = facenet(img_tensor).numpy().flatten()
                        embeddings_list.append(embedding)

                    print(f"✅ สร้าง embedding สำหรับ: {person_name}, รูป: {filename}")
                else:
                    print(f"❌ ไม่พบใบหน้าในภาพ: {image_path}")

        # ใช้ค่าเฉลี่ยของ embeddings ในการสร้างฐานข้อมูล
        if embeddings_list:
            face_embeddings[person_name] = np.mean(embeddings_list, axis=0)

# บันทึกฐานข้อมูลลงไฟล์ .npy
np.save("face_database.npy", face_embeddings)
print("📂 บันทึกฐานข้อมูลใบหน้าเสร็จสิ้น! พบทั้งหมด", len(face_embeddings), "คน")