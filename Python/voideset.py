import cv2
import os

# 视频抽帧函数
def extract_frames(video_path, output_folder, frame_interval):
    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否打开成功
    if not cap.isOpened():
        print("无法打开视频文件")
        return

    frame_count = 0  # 当前帧计数器
    saved_frame_count = 0  # 保存的帧数计数器

    while True:
        # 逐帧读取视频
        ret, frame = cap.read()

        if not ret:
            break  # 如果没有帧可读取，退出循环

        # 按照指定的间隔保存帧
        if frame_count % frame_interval == 0:
            # 保存帧到输出文件夹
            frame_filename = os.path.join(output_folder, f"frame_{saved_frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"保存帧：{frame_filename}")
            saved_frame_count += 1

        frame_count += 1

    # 释放视频文件资源
    cap.release()
    print("帧提取完成！")

# 示例调用
num = '15'
video_path = num+'.mp4'  # 视频文件路径
output_folder = './data2/'+num  # 保存帧的文件夹
frame_interval = 1  # 每隔30帧保存一张图片

extract_frames(video_path, output_folder, frame_interval)
