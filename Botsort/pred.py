from ultralytics import YOLO
import torch
import cv2
from PIL import Image
import os
import json

# model = YOLO('yolov8n.pt')
# model = YOLO('yolov8s.pt')
model = YOLO("yolov8x.pt")

output_video_path = "/Users/puw/Workspace/Vscode_Python/Bot-sort/data/res.mp4"
output_json_path = "/Users/puw/Workspace/Vscode_Python/Bot-sort/res/boxes.json"
video_path = r"/Users/puw/Workspace/Vscode_Python/Bot-sort/data/mydata.mp4"
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 使用MP4编码器
# res = cv2.VideoWriter(
#     output_video_path, fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (width, height)
# )

# for storing the track result
res_boxes = {}

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        current_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(
            frame,
            persist=True,
            show=False,
            tracker="botsort.yaml",
            device=torch.device("mps"),
        )

        # image 相关----------------
        # output_frame_path = (
        #     "/Users/puw/Workspace/Vscode_Python/Bot-sort/data/"
        #     + str(current_frame_number)
        #     + "/"
        # )
        # im_array = results.plot()
        # im = Image.fromarray(im_array[..., ::-1])
        # im.show()  # show image
        # if not os.path.exists(output_frame_path):
        #     os.makedirs(output_frame_path)
        # im.save(
        #     output_frame_path + str(current_frame_number) + ".jpg"
        # )  # save image
        # --------------------------

        if len(results) != 1:
            res_boxes[current_frame_number] = "results不为0"
        else:
            # boxes = results[0].boxes
            res_boxes[current_frame_number] = json.loads(results[0].tojson())

        # res_boxes[current_frame_number] =

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        # cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # res.write(annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # for test
        # if current_frame_number==10:
        #     break
    else:
        # Break the loop if the end of the video is reached
        break

# store res to json format
with open(output_json_path, "w") as f:
    json.dump(res_boxes, f, indent=4)

cap.release()
# res.release()
cv2.destroyAllWindows()
