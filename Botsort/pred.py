from ultralytics import YOLO
import torch
import cv2
from PIL import Image
import os
import json
import numpy as np
from scipy.stats import skew, kurtosis
from skimage.color import rgb2gray
from skimage.filters.rank import entropy
from skimage.morphology import disk
from sklearn.decomposition import PCA
import os

def track(
    video_path, output_path="", start_time=0, end_time=-1, frame_extract=3, model_type="x"
):
    model_supported = ["x", "n", "s"]
    if model_type not in model_supported:
        print('Model not supported.')
    elif model_supported == 'n':
        model = YOLO('yolov8n.pt')
    elif model_type=='s':
        model = YOLO('yolov8s.pt')
    elif model_type=='x':
        model = YOLO("yolov8x.pt")
    else:
        print('Cannot crate model.')
    video_path = video_path
    start_time = start_time
    end_time = end_time
    frame_extract = frame_extract
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video!")
        return
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    tot_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 使用MP4编码器
    start_frame = start_time * fps
    if end_time == -1:
        end_frame = tot_frame
    else:
        end_frame = end_time * fps
    output_path = (
        output_path
        + "tracked_res/chase_1_left_half_"
        + str(start_frame)
        + "_"
        + str(end_frame)
        + "/"
    )
    if not os.path.exists(output_path):
        os.makedirs(output_path + "annotated_img")
        os.makedirs(output_path + "img")
    res_video = cv2.VideoWriter(
        output_path + "res.mp4", fourcc, round(fps / frame_extract), (width, height)
    )
    # for storing the track result
    res_boxes = {}
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            current_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if current_frame_number < start_frame:
                continue
            if current_frame_number % frame_extract != 0:
                continue
            if current_frame_number > end_frame:
                break

            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(
                frame,
                persist=True,
                show=False,
                tracker="botsort.yaml",
                device=torch.device("mps"),
            )
            res_t = []
            res_img = []

            im_array = results[0].orig_img
            boxes = results[0].boxes.xyxy
            xywh_t = results[0].boxes.xywh
            shape = results[0].orig_shape

            for i in range(0, len(boxes)):
                id = results[0].boxes.id[i]
                cls = results[0].boxes.cls[i]
                xmin = int(boxes[i][0].item())
                ymin = int(boxes[i][1].item())
                xmax = int(boxes[i][2].item())
                ymax = int(boxes[i][3].item())

                target = im_array[ymin:ymax, xmin:xmax]
                rgb_mean = np.mean(target, axis=(0, 1))

                min_values = np.min(target, axis=(0, 1))
                max_values = np.max(target, axis=(0, 1))
                data_oned = np.reshape(target, (-1, 3))

                rgb_range = max_values - min_values

                rgb_var = np.var(target, axis=(0, 1))
                xywh = xywh_t[i].tolist()
                area = xywh[2] * xywh[3]
                xyxy = boxes[i].tolist()
                skewness = skew(data_oned, axis=0)
                kurt = kurtosis(data_oned, axis=0)
                gray_img = rgb2gray(target)
                img_entropy = np.mean(
                    entropy(gray_img, disk(5))
                )  # 使用一个5像素的圆盘结构元素
                res_img.append(target)

                t = {
                    "class": cls.item(),
                    "id": id.item(),
                    "rgb_mean": rgb_mean.tolist(),
                    "rgb_range": rgb_range.tolist(),
                    "rgb_var": rgb_var.tolist(),
                    "yxhw": xywh,
                    "area": area,
                    "yxyx": xyxy,
                    "skewness": skewness.tolist(),
                    "kurt": kurt.tolist(),
                    "ebtropy": img_entropy,
                }
                res_t.append(t)
            cv2.imwrite(
                output_path + "img/" + str(current_frame_number) + ".png",
                im_array,
            )

            res_boxes[current_frame_number] = res_t

            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            cv2.imwrite(
                output_path + "annotated_img/" + str(current_frame_number) + ".png",
                annotated_frame,
            )

            res_video.write(annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # store results
    with open(output_path + "boxes.json", "w") as f:
        json.dump(res_boxes, f, indent=4)
    cap.release()
    res_video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    current_directory = os.getcwd()
    track(
        "../data/left_half.mp4",#note time limit out
        start_time=0,
        end_time=5,
    )