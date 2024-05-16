from explain import make_interface


if __name__ == "__main__":
    """Main function for yielding the GUI window and explanations.

    Args:
        argument 1: image path
        argument 2: number of inter-objects occlusion 
        argument 3 (optional): the scale in which you want your image to show on the screen
        argument 4 (optional): SHAP plot type to be generated
    """
    make_interface()
