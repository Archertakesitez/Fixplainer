import PIL.Image
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import shap
import numpy as np
import pandas as pd
import os
from Botsort.get_feature import get_features_single
import PIL

#tested!
def make_SHAP(yxyx:list[float], image:PIL.Image.Image, occlusion:int)->None:
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
    ret_df = get_features_single(single_img=image, yxyx = yxyx)
    #ret_df.drop(['frame','cls'], axis = 1, inplace = True)
    ret_df['inter_objects_occlusion'] = occlusion
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
    