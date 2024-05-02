import json
import numpy as np
from scipy.stats import skew, kurtosis
from skimage.color import rgb2gray
from skimage.filters.rank import entropy
from skimage.morphology import disk
from sklearn.decomposition import PCA
import cv2
import pandas as pd

res = pd.read_csv(
    "/Users/puw/Workspace/Vscode_Python/Bot-sort/res/data_feature1.csv", index_col=0
)
# print(r)

with open("/Users/puw/Workspace/Vscode_Python/Bot-sort/data/bycicle.json", "r") as f:
    # 从文件中加载 JSON 数据并解析为 Python 字典
    data = json.load(f)
# res = pd.DataFrame()
for key, value in data.items():
    for v in value:
        if "class" in v.keys() and v["class"] == 1:
            if "id" not in v.keys():
                image = cv2.imread(
                    "/Users/puw/Workspace/Vscode_Python/Bot-sort/res/bicycle/"
                    + str(key)
                    + ".png"
                )
                xmin = int(v["yxyx"][0])
                ymin = int(v["yxyx"][1])
                xmax = int(v["yxyx"][2])
                ymax = int(v["yxyx"][3])
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

                r_mean = rgb_mean[0]
                g_mean = rgb_mean[1]
                b_mean = rgb_mean[2]
                r_range = rgb_range[0]
                g_range = rgb_range[1]
                b_range = rgb_range[2]
                r_var = rgb_var[0]
                g_var = rgb_var[1]
                b_var = rgb_var[2]
                x_average = (xmin + xmax) / 2
                y_average = (ymin + ymax) / 2
                height = ymax - ymin
                width = xmax - xmin
                area = height * width
                myentropy = np.mean(entropy(gray_img, disk(5)))
                r_skewness = skewness[0]
                g_skewness = skewness[1]
                b_skewness = skewness[2]
                r_kurt = kurt[0]
                g_kurt = kurt[1]
                b_kurt = kurt[2]
                luminance = rgb2gray(rgb_mean)

            # test code
            # print(key)
            # cv2.imshow('Image', target)
            # cv2.waitKey(0)

            # ========
            # store
        else:
            luminance = rgb2gray(v["rgb_mean"])
            r_mean = v["rgb_mean"][0]
            g_mean = v["rgb_mean"][1]
            b_mean = v["rgb_mean"][2]
            r_range = v["rgb_range"][0]
            g_range = v["rgb_range"][1]
            b_range = v["rgb_range"][2]
            r_var = v["rgb_var"][0]
            g_var = v["rgb_var"][1]
            b_var = v["rgb_var"][2]
            x_average = v["yxhw"][0]
            y_average = v["yxhw"][1]
            xmin = int(v["yxyx"][0])
            ymin = int(v["yxyx"][1])
            xmax = int(v["yxyx"][2])
            ymax = int(v["yxyx"][3])
            height = ymax - ymin
            width = xmax - xmin
            area = v["area"]
            myentropy = v["ebtropy"]
            r_skewness = v["skewness"][0]
            g_skewness = v["skewness"][1]
            b_skewness = v["skewness"][2]
            r_kurt = v["kurt"][0]
            g_kurt = v["kurt"][1]
            b_kurt = v["kurt"][2]

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
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
        }
        res = pd.concat([res, pd.DataFrame([dict])], ignore_index=True)
        # if v["class"] == 1 :
        #     image = cv2.imread(
        #         "/Users/puw/Workspace/Vscode_Python/Bot-sort/res/bicycle/" + str(key) + ".png"
        #     )
        #     target = image[ymin:ymax, xmin:xmax]
        #     cv2.imshow("Image", target)
        #     cv2.waitKey(0)
        #     # break

print(res)
# res.to_csv(
#     "/Users/puw/Workspace/Vscode_Python/Bot-sort/res/data_feature.csv", index=False
# )


cv2.destroyAllWindows()
