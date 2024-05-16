# Fixplainer: Failure Explainer for Multiple Object Tracking (MOT) --In progress!!
> [**Fixplainer: Failure Explainer for Multiple Object Tracking (MOT)**](https://drive.google.com/file/d/1EUnTP8E9paZQn6ERtoMdSVkU1R0cOo92/view?usp=sharing)
> 
> Sunny Yang, Pu Wang, Mike Pan, Erchi Zhang
<p align = "center">
      <img src="https://github.com/Archertakesitez/Fixplainer/blob/main/readme_sources/fixplainer_GUI.png" alt="FixplainerGUI" width="600"/>
</p>
Final Project for NYU's graduate course, DS-GA 3001: Visualization for Machine Learning

## Highlights ⭐️
- straight-forward GUI
- multiple YOLOv8 pretrained models support
- multiple SHAP plot type support
- functions to train on your own videos

## Abstract
Fixplainer delves into the complexities of urban dynamics by concentrating on the enhancement and understanding of
multi-object tracking (MOT) technologies within densely populated urban settings. It is a GUI tool that could generate SHAP explanation plots for the objects that the users
drawn on a video frame.

## Training Set Description
### Dataset Source: [StreetAware: A High-Resolution Synchronized Multimodal Urban Scene Dataset](https://drive.google.com/drive/u/1/folders/1BPtiIF8gBOoZANAGkwDjJUYakpCUYHM1)
The StreetAware dataset is a high-resolution synchronized multimodal urban scene dataset containing more than 8 hours of recordings of busy intersections in Brooklyn, NY.
## Getting Started 🚀
### Installation
**1.** Set up with conda environment:
```
conda create --name fixplainer_env python=3.9
conda activate fixplainer_env
```

**2.** Install Fixplainer:
```
git clone git@github.com:Archertakesitez/Fixplainer.git
cd Fixplainer
pip install -r requirements.txt
```

Then you are ready to go!

### Demo
**1.** 
Put your image inside run/ folder.
```
<Fixplainer>
      │ 
      └── <run>
            │ 
            └── {your_image}
```

**2.** 
Execute main.py, write the **first argument** as your image path, the **second argument** as the inter-objects occlusion value, i.e., the number of objects to be tracked that is overlapped inside the box you will be plotting, the **third argument** (optional) as the scale you want your image to be shown in your screen, and the **fourth argument** (optional) as the SHAP plot type you want to generate (either "waterfall" or "decision"). For example, if you want to analyze test1.png, where no detected objects are present inside the box area you will be drawing, you can run:
```
cd run
python main.py test1.png 0
```
   If you want to analyze test.png, where one detected objects are present inside the box area you will be drawing, and you want to scale the image down to 0.5*its original size, with decision plot generating, you can run:
```
python main.py test.png 1 --scale 0.5 --plot_type decision
```
Then the GUI window will show in the scale you acquired.

**3.** Inside the GUI window, you can use your mouse to draw boxes for any objects for analyzing, and the boxes you draw would appear to have thick white dotted lines:
<p align="center">
  <img src="https://github.com/Archertakesitez/Fixplainer/blob/main/readme_sources/example2.png" alt="example2" width="600"/>
  <img src="https://github.com/Archertakesitez/Fixplainer/blob/main/readme_sources/example1.png" alt="example1" width="600"/>
</p>

**4.** As soon as you release your mouse after drawing the box, the SHAP plot analyzing your selected object will appear:
<p align = "center">
      <img src="https://github.com/Archertakesitez/Fixplainer/blob/main/readme_sources/example3.png" alt="example3" width="600"/>
</p>

### Additional Functions
Besides using our pre-trained models for generating SHAP plots to analzye your images, you can use our functions to train on your own dataset!



## Authors 🔥
- **[Sunny Yang](https://github.com/crimsonsunny22)**
- **[Pu Wang](https://github.com/Puw242)**
- **[Mike Pan](https://github.com/Leo10101010)**
- **[Erchi Zhang](https://github.com/Archertakesitez)**
