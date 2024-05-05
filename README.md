# Fixplainer: Failure Explainer for Multiple Object Tracking (MOT)
> Sunny Yang, Pu Wang, Mike Pan, Erchi Zhang

Final Project for NYU's graduate course, DS-GA 3001: Visualization for Machine Learning
## Abstract
## Training Set Description
### Dataset Source: [StreetAware: A High-Resolution Synchronized Multimodal Urban Scene Dataset](https://drive.google.com/drive/u/1/folders/1BPtiIF8gBOoZANAGkwDjJUYakpCUYHM1)
The StreetAware dataset is a high-resolution synchronized multimodal urban scene dataset containing more than 8 hours of recordings of busy intersections in Brooklyn, NY.
## Getting Started 🚀
### Installation
**1.** Set up with conda environment:
```
conda create -n fixplainer_env python=3.9
conda activate fixplainer_env
```

**2.** Install Fixplainer:
```
git clone git@github.com:Archertakesitez/Fixplainer.git
cd Fixplainer
pip3 install -r requirements.txt
```

Then you are ready to go!

### Demo
**1.** Put your image inside Fixplainer folder.
```
<Fixplainer>
      │ 
      └── {your_image}
```

**2.** Execute main.py, write the **first argument** as your image name, the _second argument_ as the inter-objects occlusion value, i.e., the number of objects to be tracked that is overlapped inside the box you will be plotting, and the _third argument_ (optional) as the scale you want your image to be shown in your screen. For example, if you want to analyze test.png, where no detected objects are present inside the box area you will be drawing, you can run:
```
cd Fixplainer
python3 main.py test.png 0
```
   If you want to analyze test.png, where one detected objects are present inside the box area you will be drawing, and you want to scale the image down to 0.5*its original size, you can run:
```
python3 main.py test.png 1 0.5
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

## Authors 🔥
- **[Sunny Yang](https://github.com/crimsonsunny22)**
- **[Pu Wang](https://github.com/Puw242)**
- **[Mike Pan](https://github.com/Leo10101010)**
- **[Erchi Zhang](https://github.com/Archertakesitez)**
