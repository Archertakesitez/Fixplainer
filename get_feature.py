import json
import numpy as np
from scipy.stats import skew, kurtosis
from skimage.color import rgb2gray
from skimage.filters.rank import entropy
from skimage.morphology import disk
from sklearn.decomposition import PCA
import cv2
import pandas as pd
import os
from PIL import Image


def get_features(
    labeled_data_path="",
    image_path="",
    old_data_path="",
    output_path="",
    save=True,
    xyxy=[],
):
    im_w = 0
    im_h = 0
    if old_data_path != "":
        res = pd.read_csv(
            old_data_path,
            index_col=0,
        )
    else:
        res = pd.DataFrame()
    output_path = output_path
    if output_path != "" and not os.path.exists(output_path):
        os.makedirs(output_path)

    if isinstance(image_path, Image.Image):
        data = {-1: [{"yxyx": xyxy, "class": 1}]}
        im_h, im_w = image_path.size

    else:
        with open(labeled_data_path, "r") as f:
            # 从文件中加载 JSON 数据并解析为 Python 字典
            data = json.load(f)
        files = os.listdir(image_path)
        for file in files:
            if file.endswith(".png"):
                im_h, im_w, _ = cv2.imread(image_path + "/" + file).shape
                break

    for key, value in data.items():
        for v in value:
            if "class" in v.keys() and v["class"] == 1:
                if "id" not in v.keys():
                    if key == -1:
                        image = cv2.cvtColor(np.array(image_path), cv2.COLOR_RGB2BGR)
                    else:
                        image = cv2.imread(image_path + "/" + str(key) + ".png")
                    xmin = int(v["yxyx"][0])
                    ymin = int(v["yxyx"][1])
                    xmax = int(v["yxyx"][2])
                    ymax = int(v["yxyx"][3])
                    height = (ymax - ymin) / im_h
                    width = (xmax - xmin) / im_w
                    target = image[ymin:ymax, xmin:xmax]
                    rgb_mean = np.mean(target, axis=(0, 1))
                    min_values = np.min(target, axis=(0, 1))
                    max_values = np.max(target, axis=(0, 1))
                    data_oned = np.reshape(target, (-1, 3))
                    rgb_range = max_values - min_values
                    rgb_var = np.var(target, axis=(0, 1))
                    skewness = skew(data_oned, axis=0)
                    kurt = kurtosis(data_oned, axis=0)
                    gray_img = rgb2gray(target)

                    b_mean = rgb_mean[0]
                    g_mean = rgb_mean[1]
                    r_mean = rgb_mean[2]
                    b_range = rgb_range[0]
                    g_range = rgb_range[1]
                    r_range = rgb_range[2]
                    b_var = rgb_var[0]
                    g_var = rgb_var[1]
                    r_var = rgb_var[2]
                    x_average = (xmin + xmax) / (2 * im_w)
                    y_average = (ymin + ymax) / (2 * im_h)
                    area = height * width / im_h / im_w
                    myentropy = np.mean(entropy(gray_img, disk(5)))
                    b_skewness = skewness[0]
                    g_skewness = skewness[1]
                    r_skewness = skewness[2]
                    b_kurt = kurt[0]
                    g_kurt = kurt[1]
                    r_kurt = kurt[2]
                    luminance = rgb2gray(rgb_mean)

                # test code
                # print(key)
                # cv2.imshow('Image', target)
                # cv2.waitKey(0)

                # ========
                # store
            else:
                luminance = rgb2gray(v["rgb_mean"])
                b_mean = v["rgb_mean"][0]
                g_mean = v["rgb_mean"][1]
                r_mean = v["rgb_mean"][2]
                b_range = v["rgb_range"][0]
                g_range = v["rgb_range"][1]
                r_range = v["rgb_range"][2]
                b_var = v["rgb_var"][0]
                g_var = v["rgb_var"][1]
                r_var = v["rgb_var"][2]
                xmin = int(v["yxyx"][0])
                ymin = int(v["yxyx"][1])
                xmax = int(v["yxyx"][2])
                ymax = int(v["yxyx"][3])
                height = (ymax - ymin) / im_h
                width = (xmax - xmin) / im_w
                x_average = v["yxhw"][0] / im_w
                y_average = v["yxhw"][1] / im_h
                area = v["area"] / im_w / im_h
                myentropy = v["ebtropy"]
                b_skewness = v["skewness"][0]
                g_skewness = v["skewness"][1]
                r_skewness = v["skewness"][2]
                b_kurt = v["kurt"][0]
                g_kurt = v["kurt"][1]
                r_kurt = v["kurt"][2]

            dict = {
                "frame": key,
                "cls": int(v["class"]),
                "r_mean": r_mean,
                "g_mean": g_mean,
                "b_mean": b_mean,
                "r_range": r_range,
                "g_range": g_range,
                "b_range": b_range,
                "r_var": r_var,
                "g_var": g_var,
                "b_var": b_var,
                "x_average": x_average,
                "y_average": y_average,
                "height": height,
                "width": width,
                "area": area,
                "entropy": myentropy,
                "r_skewness": r_skewness,
                "g_skewness": g_skewness,
                "b_skewness": b_skewness,
                "r_kurtosis": r_kurt,
                "g_kurtosis": g_kurt,
                "b_kurtosis": b_kurt,
                "luminance": luminance,
                "xmin": xmin / im_w,
                "xmax": xmax / im_w,
                "ymin": ymin / im_h,
                "ymax": ymax / im_h,
            }
            res = pd.concat([res, pd.DataFrame([dict])], ignore_index=True)

            # show wrong tracked boxes
            # if v["class"] == 1 :
            #     image = cv2.imread(
            #         "/Users/puw/Workspace/Vscode_Python/Bot-sort/res/bicycle/" + str(key) + ".png"
            #     )
            #     target = image[ymin:ymax, xmin:xmax]
            #     cv2.imshow("Image", target)
            #     cv2.waitKey(0)
            #     # break

    if save:
        res.to_csv(output_path + "data_feature.csv", index=False)
    cv2.destroyAllWindows()
    return res


if __name__ == "__main__":
    get_features(
        labeled_data_path="/Users/puw/Workspace/Vscode_Python/Bot-sort/res/id4.json",
        image_path="/Users/puw/Workspace/Vscode_Python/Bot-sort/tracked_res/chase_1_left_half_9520_9632/img",
    )
    # pil_image = Image.open(
    #     "/Users/puw/Workspace/Vscode_Python/Bot-sort/res/img/9522.png"
    # )

