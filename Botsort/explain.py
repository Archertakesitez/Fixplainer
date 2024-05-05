import tkinter as tk
from PIL import Image, ImageTk
from Botsort.produce_plot import make_SHAP
import sys

class ImageSelector:
    """
    This class builds a tool that allows user to select boxes inside their uploaded image,
    and yield SHAP explanation for while their selected box cannot or can be successfully tracked
    by YOLOv8+BoT-SORT
    """
    def __init__(self, root, image_path:str, occlusion:int, scale=1)->None:
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
        self._init_ui()

    def _init_ui(self):
        """
        Initialze the image selector GUI.
        """
        original_image = Image.open(self.image_path)
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
        print(f"Coordinates stored: Top-Left ({self.topx}, {self.topy}) Bottom-Right ({self.botx}, {self.boty})")
        print(f"{self.img.width()}, {self.img.height()}")
        make_SHAP(xyxy = [self.topx, self.topy, self.botx, self.boty], image = self.resized, occlusion = self.occlusion)
        

def make_interface():
    """
    This function yields the GUI window for user to plot box inside their uploaded image.

    Args:
        arguments 1: image path
        arguments 2 (optional): the scale you want the image to show on your screen
    """
    if len(sys.argv) == 1 or len(sys.argv) == 2:
        print(len(sys.argv))
        print("please provide image path and occlusion!")
    elif len(sys.argv) == 3:
        image_path = sys.argv[1]
        occlusion = sys.argv[2]
        root = tk.Tk()
        app = ImageSelector(root, image_path=image_path, occlusion = int(occlusion))
        root.mainloop()
    elif len(sys.argv) == 4:
        image_path = sys.argv[1]
        occlusion = sys.argv[2]
        scale = sys.argv[3]
        root = tk.Tk()
        app = ImageSelector(root, image_path=image_path, occlusion = int(occlusion), scale = scale)
        root.mainloop()


