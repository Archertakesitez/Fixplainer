# Fixplainer: Explaining Failure/Success Patterns in Multiple Object Tracking (MOT)
> Sunny Yang, Pu Wang, Mike Pan, Erchi Zhang

Final Project for NYU's graduate course, DS-GA 3001: Visualization for Machine Learning
## Abstract
## Data Description
### Dataset Source: [StreetAware: A High-Resolution Synchronized Multimodal Urban Scene Dataset](https://drive.google.com/drive/u/1/folders/1BPtiIF8gBOoZANAGkwDjJUYakpCUYHM1)
The StreetAware dataset is a high-resolution synchronized multimodal urban scene dataset containing more than 8 hours of recordings of busy intersections in Brooklyn, NY.
## Getting Started
### Installation
### Demo
1. Put any image in which you want to see SHAP explanations for failed/successful tracking inside Fixplainer folder.
2. go inside Fixplainer folder:
```
cd Fixplainer
```
3. Execute main.py, write the first argument as your image name, the second argument (optional) as the scale you want your image to be shown in your screen. For example, if you want to analyze test.png, you can run:
```
python3 main.py test.png
```
If you want to analyze test.png and scale it down to 0.5*its original size, you can run:
```
python3 main.py test.png 0.5
```
Then the GUI window will show in the scale you acquired.
4. Inside the GUI window, you can use your mouse to draw boxes for any objects for analyzing, and the boxes you draw would appear to have white thick dotted lines:
![example1](https://github.com/Archertakesitez/Fixplainer/blob/main/readme_sources/example1.png)
## Authors
- **[Sunny Yang](https://github.com/crimsonsunny22)**
- **[Pu Wang](https://github.com/Puw242)**
- **[Mike Pan](https://github.com/Leo10101010)**
- **[Erchi Zhang](https://github.com/Archertakesitez)**
