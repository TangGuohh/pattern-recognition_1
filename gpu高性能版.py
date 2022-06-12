import cv2
import numpy
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cupy as cp

def cv2ImgAddText(img, text, left, top, textColor=(0, 0, 255), textSize=40):
    if (isinstance(img, numpy.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)

video = "./demo.mp4"
result_video = "./demo-result.mp4"
#读取视频
cap = cv2.VideoCapture(video)
#获取视频帧率
fps_video = cap.get(cv2.CAP_PROP_FPS)
fps_video = fps_video - 5
#设置写入视频的编码格式
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#获取视频宽度
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#获取视频高度
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
videoWriter = cv2.VideoWriter(result_video, fourcc, fps_video, (frame_width, frame_height))
frame_num = 0
print(frame_width,frame_height)
frame_num_sum = 916
centerX = frame_height / 2
centerY = frame_width / 2
radius = 10
strength = 150
distance = []
for i in range(frame_height):
    temp_list = []
    for j in range(frame_width):
        # 计算当前点到光照中心距离(平面坐标系中两点之间的距离)
        temp = np.power((centerY - j), 2) + np.power((centerX - i), 2)
        temp_list.append(temp)
    distance.append(temp_list)
distance = np.array(np.sqrt(distance))
while (cap.isOpened()):
    ret, frame = cap.read()
    frame_num += 1
    print("当前处理第{}帧".format(frame_num))

    # # 参数ret为True或者False, 代表有没有读取到图片
    # # 第二个参数frame表示截取到一帧的图片
    if (ret == True) and (frame_num<=fps_video*5):
        #文字坐标
        # cv2.putText(frame, "听我说谢谢你" , (word_x, word_y),cv2.FONT_HERSHEY_SIMPLEX, 1, (55,255,155), 2)
        # cv2.putText(frame, "why are you so diao", (605, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        frame = cv2ImgAddText(frame, "听我说谢谢你", 505, 650)
        # cv2.imshow('hh', frame)
        # cv2.waitKey(0)
        # print(frame.shape)
        videoWriter.write(frame)

    elif (ret == True) and (frame_num<=fps_video*15):
        frame = cv2ImgAddText(frame, "因为有你温暖了四季", 505, 650)
        videoWriter.write(frame)
    #     最后一帧应该为：916
    elif (ret == True) and (frame_num <= fps_video*20):
        videoWriter.write(frame)
    elif (ret == True) and (frame_num <= fps_video*24):
        # frame[:, :,0] =
        radius = radius + 10
        strength = strength + 2
        for i in range(frame_height):
            for j in range(frame_width):
                # 计算当前点到光照中心距离(平面坐标系中两点之间的距离)
                # 获取原始图像
                if (distance[i][j] < radius):
                    # 按照距离大小计算增强的光照值
                    result = (int)(strength * (1.0 - distance[i][j] / radius))
                    B = frame[i, j][0] + result
                    G = frame[i, j][1] + result
                    R = frame[i, j][2] + result
                    # 判断边界 防止越界
                    B = min(255, max(0, B))
                    G = min(255, max(0, G))
                    R = min(255, max(0, R))
                    frame[i, j] = np.uint8((B, G, R))
        videoWriter.write(frame)
    elif (ret == True) and (frame_num<=fps_video*28):
        # frame[:, :,0] =
        radius = radius - 10
        strength = strength - 2
        for i in range(frame_height):
            for j in range(frame_width):
                # 计算当前点到光照中心距离(平面坐标系中两点之间的距离)
                # 获取原始图像
                if (distance[i][j] < radius):
                    # 按照距离大小计算增强的光照值
                    result = (int)(strength * (1.0 - distance[i][j] / radius))
                    B = frame[i, j][0] + result
                    G = frame[i, j][1] + result
                    R = frame[i, j][2] + result
                    # 判断边界 防止越界
                    B = min(255, max(0, B))
                    G = min(255, max(0, G))
                    R = min(255, max(0, R))
                    frame[i, j] = np.uint8((B, G, R))
        videoWriter.write(frame)
    elif (ret == True) and (frame_num <= 900):
        videoWriter.write(frame)
    else:
        videoWriter.release()
        break
