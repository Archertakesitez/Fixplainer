from explain import make_interface


if __name__ == "__main__":
    """Main function for yielding the GUI window and explanations.

    Args:
        argument 1: image path
        argument 2: number of inter-objects occlusion 
        --scale (optional): the scale in which you want your image to show on the screen
        --plot_type (optional): SHAP plot type to be generated
        --model_path (optional): the pretrained model you want to use
        --X_train_path (optional): the X_train set corresponding to your pretrained model
    """
    make_interface()
