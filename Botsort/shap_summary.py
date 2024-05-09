import matplotlib.pyplot as plt
import pickle
import shap
import os

def get_summary()->None:
    current_directory = os.getcwd()#fetch current repository
    with open(current_directory+'/pretrained_tools/pretrained_xgboost.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    with open(current_directory+'/pretrained_tools/X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    explainer = shap.Explainer(loaded_model)
    shap_values = explainer(X_train)
    max_display = len(X_train.columns.unique())
    shap.summary_plot(shap_values, X_train, show = False, max_display = max_display)
    plt.title(f"SHAP summary plot")
    plt.gcf().set_size_inches(6, 6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    get_summary()