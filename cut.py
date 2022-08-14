import os
import cv2 as cv
import argparse
import numpy as np
import cv2

weightsPath = "yolov3.weights"
configPath = "yolov3.cfg"
rootdir = r"E:\FFOutput\shiju\20210524083714_20210524090000_420787"  # 图像读取地址
savepath = "E:\FFOutput\shiju\output"  # 图像保存地址

# 初始化一些参数

filelist = os.listdir(rootdir)  # 打开对应的文件夹
total_num = len(filelist)  # 得到文件夹中图像的个数print(total_num)
# 如果输出的文件夹不存在，创建即可
if not os.path.isdir(savepath):
    os.makedirs(savepath)

for (dirpath, dirnames, filenames) in os.walk(rootdir):
    for filename in filenames:
        # 必须将boxes在遍历新的图片后初始化
        boxes = []
        confidences = []
        classIDs = []
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        path = os.path.join(dirpath, filename)
        image = cv.imread(path)
        print(path)
        (H, W) = image.shape[:2]
        # 得到 YOLO需要的输出层
        ln = net.getLayerNames()
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
        # 从输入图像构造一个blob，然后通过加载的模型，给我们提供边界框和相关概率
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)
        # 在每层输出上循环
        for output in layerOutputs:
            # 对每个检测进行循环
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # 过滤掉那些置信度较小的检测结果
                if confidence > 0.8:
                    # 框后接框的宽度和高度
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    # 边框的左上角
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # 更新检测出来的框
                    # 批量检测图片注意此处的boxes在每一次遍历的时候要初始化，否则检测出来的图像框会叠加
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        # 极大值抑制
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.3)
        k = -1
        if len(idxs) > 0:
            # for k in range(0,len(boxes)):
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # image是原图，     左上点坐标， 右下点坐标， 颜色， 画线的宽度
                cv2.rectangle(image, (x, y), (x + w, y + h),(224, 224, 224), 2)
                # 各参数依次是：图片，添加的文字，左上角坐标(整数)，字体，        字体大小，颜色，字体粗细
                cv2.putText(image, "text", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (224, 224, 224), 2)
                # 图像裁剪注意坐标要一一对应
                # 图片裁剪 裁剪区域【Ly:Ry,Lx:Rx】
                cut = image[y:(y + h), x:(x + w)]
                if(cut.shape[0] != 0 and cut.shape[1] != 0 and cut.shape[2] != 0):
                    # 写入文件夹，这块写入的时候不支持int（我也不知道为啥），所以才用的字母
                    cv.imwrite(savepath + "/" + filename.split(".")[0] + "_b" + ".jpg", cut)
