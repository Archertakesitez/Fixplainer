import PIL.Image
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import shap
import numpy as np
import pandas as pd
import os
from Botsort.get_feature import get_features
import PIL


def make_SHAP(xyxy:list[float], image:PIL.Image.Image, occlusion:int)->None:
    """
    This function aims to extract features from the box plotted by the user in GUI,
    and use the pretrained XGBoost model to make prediction for whether the object in the box could 
    be successfully tracked or not, and then use SHAP waterfall plot to explain which feature
    contributes to the failure/success of tracking.
    
    Args:
        image_width: width of the image uploaded
        image_height: height of the image uploaded
        topx: x-coordinate of the box's top left corner
        topy: y-coordinate of the box's top left corner
        botx: x-coordinate of the box's bottom right corner
        boty: y-coordinate of the box's bottom right corner
    """
    current_directory = os.getcwd()#fetch current repository
    with open(current_directory+'/Botsort/pretrained_tools/pretrained_xgboost.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    ret_df = get_features(image_path=image, xyxy = xyxy, save = True)
    ret_df.drop(['frame','cls'], axis = 1, inplace = True)
    # Convert dictionary to DataFrame
    index = ['r_mean', 'g_mean', 'b_mean', 'r_range', 'g_range', 'b_range', 'r_var',
       'g_var', 'b_var', 'x_average', 'y_average', 'height', 'width', 'area',
       'entropy', 'r_skewness', 'g_skewness', 'b_skewness', 'r_kurtosis',
       'g_kurtosis', 'b_kurtosis', 'luminance', 'xmin', 'xmax', 'ymin',
       'ymax', 'occlusion']
    #X_new = pd.DataFrame(ret_df,index=index)
    ret_df['occlusion'] = occlusion
    with open(current_directory+'/Botsort/pretrained_tools/X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    explainer = shap.Explainer(loaded_model,X_train)
    label = loaded_model.predict(ret_df)[0]
    output = "successful" if label == 0 else "failed"
    shap_values = explainer(ret_df)
    shap.plots.waterfall(shap_values[0],show=False)
    plt.title(f"SHAP expalanations for this {output} tracking")
    plt.gcf().set_size_inches(8, 4)
    plt.tight_layout()
    plt.show()
    