import tkinter as tk
from PIL import Image, ImageTk
from produce_plot import make_SHAP
import sys
import argparse

#tested!
class ImageSelector:
    """
    This class builds a tool that allows user to select boxes inside their uploaded image,
    and yield SHAP explanation for while their selected box cannot or can be successfully tracked
    by YOLOv8+BoT-SORT
    """
    def __init__(self, root, image_path:str, occlusion:int, scale=1, plot_type = "waterfall", model_path = "pretrained_tools/pretrained_xgboost.pkl", X_train_path = "pretrained_tools/X_train.pkl")->None:
        """
        set args and call _init_ui.

        Args:
            root: root window for tkinter
            image_path: image name that user wants to analyze
            scale: the scale you want your image to be resized in
        """
        self.root = root
        self.image_path = image_path
        self.topx, self.topy, self.botx, self.boty = 0, 0, 0, 0
        self.rect_id = None
        self.scale = float(scale)
        self.root.title("Fixplainer")
        self.occlusion = occlusion
        self.plot_type = plot_type
        self.model_path = model_path
        self.X_train_path = X_train_path
        self._init_ui()

    def _init_ui(self):
        """
        Initialze the image selector GUI.
        """
        try:
            original_image = Image.open(self.image_path)
        except FileNotFoundError:
            print("Please enter a valid image path!")
            sys.exit()
        self.width = original_image.width
        self.height = original_image.height
        resized_image = original_image.resize((int(self.width*self.scale), int(self.height*self.scale)))
        self.resized = resized_image
        self.img = ImageTk.PhotoImage(resized_image)
        self.canvas = tk.Canvas(self.root, width=self.width * self.scale, height=self.height * self.scale,
                                borderwidth=0, highlightthickness=0)
        self.canvas.pack(expand=True)
        self.canvas.create_image(0, 0, image=self.img, anchor=tk.NW)
        self.canvas.bind('<Button-1>', self.get_mouse_posn)
        self.canvas.bind('<B1-Motion>', self.update_sel_rect)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_release)


    def get_mouse_posn(self, event):
        """
        get the mouse's coordinates and draw box

        Args:
            event: the mouse click
        """
        self.topx, self.topy = event.x, event.y
        self.rect_id = self.canvas.create_rectangle(self.topx, self.topy, self.topx, self.topy,
                                                    dash=(4,4), fill='', outline='white', width=3, tags = "")

    def update_sel_rect(self, event):
        """
        update mouse coordinates while moving mouse to draw box
        
        Args:
            event: the mouse's movement
        """
        self.botx, self.boty = event.x, event.y
        self.canvas.coords(self.rect_id, self.topx, self.topy, self.botx, self.boty)

    def on_mouse_release(self, event):
        """
        update mouse coordinates when the mouse releases;
        then call make_SHAP to produce SHAP explainer plots

        Args:
            event: the mouse's release after finished drawing
        """
        self.update_sel_rect(event)
        #print(f"Coordinates stored: Top-Left ({self.topx}, {self.topy}) Bottom-Right ({self.botx}, {self.boty})")
        #print(f"{self.img.width()}, {self.img.height()}")
        make_SHAP(yxyx = [self.topx, self.topy, self.botx, self.boty], image = self.resized, occlusion = self.occlusion, plot_type = self.plot_type, model_path = self.model_path, X_train_path = self.X_train_path)
        

def make_interface():
    """
    This function yields the GUI window for user to plot box inside their uploaded image.
    Parameters explanation are included in the help messages.
    """
    parser = argparse.ArgumentParser(description='let\'s set parameters for producing plot!')
    parser.add_argument('image_path', help='image path that contains the object you want to analyze for MOT failure/success')
    parser.add_argument('occlusion', type=int, help = 'how many objects that can be detected are overlapping with the box you want to draw?')
    parser.add_argument('--scale', type = float, default = 1, help='in which scale do you want your image to be shown?')
    parser.add_argument('--plot_type', default = "waterfall", help='the shap plot you want to produce--waterfall or decision? default is waterfall plot')
    parser.add_argument('--model', default = 'pretrained_tools/pretrained_xgboost.pkl', help = 'which pretrained model you want to use')
    parser.add_argument('--X_train', default = 'pretrained_tools/X_train.pkl', help = 'X_train corresponding to your pretrained model')
    args = parser.parse_args()
    image_path = args.image_path
    occlusion = args.occlusion
    scale = args.scale
    plot_type = args.plot_type
    model_path = args.model
    X_train_path = args.X_train
    root = tk.Tk()
    app = ImageSelector(root = root, image_path = image_path, occlusion = occlusion, scale = scale, plot_type = plot_type, model_path = model_path, X_train_path = X_train_path)
    root.mainloop()

