import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import shap
import numpy as np
import pandas as pd
import os

def make_SHAP(image_width:float,image_height:float,topx:float,topy:float,botx:float,boty:float)->None:
    """This function aims to extract features from the box plotted by the user in GUI,
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
    scaled_topx = topx/image_width
    scaled_topy = topy/image_height
    scaled_botx = botx/image_width
    scaled_boty = boty/image_height
    data = {
    'r_mean': 1,
    'g_mean': 1,
    'b_mean': 1,
    'r_range': 1,
    'g_range':1,
    'b_range':1,
    'r_var':1,
    'g_var':1,
    'b_var':1,
    'x_average':1,
    'y_average':1,
    'height':1,
    'width':1,
    'area':1,
    'entropy':1,
    'r_skewness':1,
    'g_skewness':1,
    'b_skewness':1,
    'r_kurtosis':1,
    'g_kurtosis':1,
    'b_kurtosis':1,
    'luminance':1,
    'xmin':1,
    'xmax':1,
    'ymin':1,
    'ymax':1
}
    # Convert dictionary to DataFrame
    index = ['r_mean', 'g_mean', 'b_mean', 'r_range', 'g_range', 'b_range', 'r_var',
       'g_var', 'b_var', 'x_average', 'y_average', 'height', 'width', 'area',
       'entropy', 'r_skewness', 'g_skewness', 'b_skewness', 'r_kurtosis',
       'g_kurtosis', 'b_kurtosis', 'luminance', 'xmin', 'xmax', 'ymin',
       'ymax']
    X_new = pd.DataFrame(data,index=index)
    with open(current_directory+'/Botsort/pretrained_tools/X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    explainer = shap.Explainer(loaded_model,X_train)
    label = loaded_model.predict(X_new)[0]
    output = "successful" if label == 0 else "failed"
    shap_values = explainer(X_new)
    shap.plots.waterfall(shap_values[0],show=False)
    plt.title(f"SHAP expalanations for this {output} tracking")
    plt.gcf().set_size_inches(8, 4)
    plt.tight_layout()
    plt.show()
    