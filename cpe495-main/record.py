import cv2
import time
import os
from datetime import datetime

# IP ของกล้อง Tapo C320WS
camera_ip = 'rtsp://Face495:CPE495@172.20.10.5:554/stream1'

# ตั้งค่ากล้อง RTSP โดยใช้ FFmpeg หรือ GStreamer
capture = cv2.VideoCapture(camera_ip, cv2.CAP_FFMPEG)

# ตรวจสอบการเชื่อมต่อกล้อง RTSP
if not capture.isOpened():
    print("Error: ไม่สามารถเชื่อมต่อกับกล้อง RTSP โดยใช้ FFmpeg")
    print("กำลังลองเชื่อมต่อโดยใช้ GStreamer...")

    # ลองใช้ GStreamer
    capture = cv2.VideoCapture(camera_ip, cv2.CAP_GSTREAMER)

    if not capture.isOpened():
        print("Error: ไม่สามารถเชื่อมต่อกับกล้อง RTSP โดยใช้ GStreamer")
        exit()

print("✅ เชื่อมต่อกับกล้อง RTSP สำเร็จ!")

# ตั้งค่าความละเอียดที่เหมาะสม
capture.set(3, 1280)
capture.set(4, 720)

# ตั้งค่า FPS
fps = 15

# เพิ่ม buffer size สำหรับ RTSP
capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)

# โฟลเดอร์บันทึกวิดีโอ
output_folder = "recorded_videos"
os.makedirs(output_folder, exist_ok=True)

# ฟังก์ชันสร้างชื่อไฟล์วิดีโอตามเวลา
def get_filename():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(output_folder, f"record_{timestamp}.mp4")

# ตั้งค่า codec และ video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# ตั้งเวลาสำหรับตัดคลิป
clip_duration = 2 * 60  # 2 นาที
start_time = time.time()

while True:
    video_filename = get_filename()
    done_filename = video_filename + ".done"

    video_out = cv2.VideoWriter(video_filename, fourcc, fps, (1280, 720))
    
    print(f"📹 เริ่มบันทึกวิดีโอ: {video_filename}")

    while True:
        ret, frame = capture.read()
        if not ret:
            print("❌ Error: ไม่สามารถอ่านเฟรมจากกล้อง!")
            break

        video_out.write(frame)
        cv2.imshow('Recording (Face Detection)', frame)

        # ตรวจสอบเวลาที่ผ่านไปและตัดคลิปใหม่
        if time.time() - start_time >= clip_duration:
            break

        # กด 'q' เพื่อออก
        if cv2.waitKey(1) & 0xFF == ord('q'):
            capture.release()
            cv2.destroyAllWindows()
            exit()

    # ปิดไฟล์วิดีโอ
    video_out.release()
    
    # สร้างไฟล์ `.done`
    with open(done_filename, "w") as f:
        f.write("done")

    print(f"✅ บันทึกเสร็จสิ้น: {video_filename}")

    # รีเซ็ตเวลาเริ่มต้น
    start_time = time.time()
