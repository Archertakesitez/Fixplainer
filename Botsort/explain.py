import tkinter as tk
from PIL import Image, ImageTk
from produce_plot import make_SHAP
import sys

class ImageSelector:
    def __init__(self, root, image_path, scale=1):
        self.root = root
        self.image_path = image_path
        self.topx, self.topy, self.botx, self.boty = 0, 0, 0, 0
        self.rect_id = None
        self.scale = float(scale)
        self._init_ui()

    def _init_ui(self):
        original_image = Image.open(self.image_path)
        self.width = original_image.width
        self.height = original_image.height
        resized_image = original_image.resize((int(self.width*self.scale), int(self.height*self.scale)))
        self.img = ImageTk.PhotoImage(resized_image)
        self.canvas = tk.Canvas(self.root, width=self.width * self.scale, height=self.height * self.scale,
                                borderwidth=0, highlightthickness=0)
        self.canvas.pack(expand=True)
        self.canvas.create_image(0, 0, image=self.img, anchor=tk.NW)
        self.canvas.bind('<Button-1>', self.get_mouse_posn)
        self.canvas.bind('<B1-Motion>', self.update_sel_rect)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_release)


    def get_mouse_posn(self, event):
        self.topx, self.topy = event.x, event.y
        self.rect_id = self.canvas.create_rectangle(self.topx, self.topy, self.topx, self.topy,
                                                    dash=(4,4), fill='', outline='white', width=3, tags = "")

    def update_sel_rect(self, event):
        self.botx, self.boty = event.x, event.y
        self.canvas.coords(self.rect_id, self.topx, self.topy, self.botx, self.boty)

    def on_mouse_release(self, event):
        self.update_sel_rect(event)
        print(f"Coordinates stored: Top-Left ({self.topx}, {self.topy}) Bottom-Right ({self.botx}, {self.boty})")
        print(f"{self.img.width()}, {self.img.height()}")
        make_SHAP(image_width=self.width,image_height=self.height,topx=self.topx,topy=self.topy,botx=self.botx,boty=self.boty)
        

def make_interface():
    """
    arguments 1: image path
    arguments 2 (optional): the scale you want the image to show on your screen
    """
    if len(sys.argv) == 1:
        print("please provide image path!")
    elif len(sys.argv) == 2:
        image_path = sys.argv[1]
        root = tk.Tk()
        app = ImageSelector(root, image_path=image_path)
        root.mainloop()
    elif len(sys.argv) == 3:
        image_path = sys.argv[1]
        scale = sys.argv[2]
        root = tk.Tk()
        app = ImageSelector(root, image_path=image_path, scale = scale)
        root.mainloop()


