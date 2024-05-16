import cv2

input_video_pth = "/Users/puw/Workspace/Vscode_Python/Bot-sort/data/left_half.mp4"
output_video_path = "/Users/puw/Workspace/Vscode_Python/Bot-sort/data/mydata.mp4"

myVideo = cv2.VideoCapture(input_video_pth)

fps = int(myVideo.get(cv2.CAP_PROP_FPS))
width = int(myVideo.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(myVideo.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

print("fps:", fps)
print("width:", width)
print("height:", height)

out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

start_time = 780
end_time = 810
start_frame = int(fps * start_time)
end_frame = int(fps * end_time)

myVideo.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

while myVideo.isOpened():
    ret, frame = myVideo.read()
    if not ret:
        break

    current_frame_number = int(myVideo.get(cv2.CAP_PROP_POS_FRAMES)) - 1

    if current_frame_number < start_frame:
        continue

    if current_frame_number > end_frame:
        break
    out.write(frame)

myVideo.release()
out.release()
cv2.destroyAllWindows()