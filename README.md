# Fixplainer: Failure Explainer for Multiple Object Tracking (MOT)
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
Fixplainer delves into the complexities of urban dynamics by concentrating on the understanding of
multi-object tracking (MOT) technologies within densely populated urban settings. It is a GUI tool that could generate SHAP explanation plots for the objects that the users
drawn on a video frame they uploaded.

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

### Demo - Main Function
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
Execute main.py, write the **first argument** as your image's path, the **second argument** as the inter-objects occlusion value, i.e., the number of objects to be tracked that is overlapped inside the box you will be plotting, the **--scale argument** (optional) as the scale you want your image to be shown in your screen, and the **--plot_type argument** (optional) as the SHAP plot type you want to generate (either "waterfall" or "decision"). If you want to use your own pretrained model, you need to include the **--model argument** (optional) as the path for your pretrained model as well as the **--X_train argument** (optional) as the path for your X_train corresponding to your pretrained model. For example, if you want to analyze test1.png, where no detected objects are present inside the box area you will be drawing, you can run:
```
cd run
python main.py test1.png 0
```
   If you want to analyze test1.png, where one detected objects are present inside the box area you will be drawing, and you want to scale the image down to 0.5*its original size, with decision plot generating, you can run:
```
python main.py test1.png 1 --scale 0.5 --plot_type decision
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

## Customer Training 👑
Besides using our pre-trained models for generating SHAP plots to analzye your images, you can use our functions to train on your own dataset(videos) and annotate your own data. This allows you to prepare specific models to better analyze specific urban scenes, or even other type of scenes.
### 1.Generate Data to be Annotated From Your Video
gen_box.py allows you to upload your own videos for training purposes, and will generate Json file for you annotate your own data.

Execute gen_box.py, write the **first argument** as your video's path, the **second argument** as the output path for your Json file to be generated for annotation, the **--start_time argument** (optional) as the second you want to begin truncating your video, the **--end_time argument** (optional) as the second you want to finish truncating your video, the **--frame_extract argument** (optional) as the frequency you want to save each video frame, i.e., in how many video frames you do want to save one frame, and the **--model_type argument** (optional) as the YOLO pretrained object detector (select from 'x', 'n', 's') you want to apply on your video.  

For example, if you have uploaded a video with the path run/video.mp4, and you want to output the generated Json file for annotation to a directory output/, saving a video frame for each three video frames, you can run:
```
cd run
python gen_box.py video.mp4 output/ --frame_extract 3
```
And then you can check your Json file generated for your video.mp4 under the output/ folder. This folder will also contain a directory that has all the video frames extracted.
### 2.Annotate Data
Please refer to our [annotation guideline](https://drive.google.com/file/d/1ddUVbHqSW4ltW6E9JS7_4TlPJyxsb5t_/view?usp=sharing)!
### 3.Save Your Own Model
After you have annotated your Json file generated by gen_box.py, you could proceed to this step to train and save your own model. **train_your_own.py** handles this for you well.

Execute train_your_own.py, write the **first argument** as the directory containing all the video frames corresponding to your annotated data, write the **second argument** as the path for the Json file you have annotated, write the **--old_path argument** (optional) as the csv file that you have generated before and in which you would like to add new annotated rows on, and write the **--save argument** (optional) if you want to save the output (features of objects in your video) to a csv file. 

For example, if you want have followed step 1 and annotated Json file as output/labeled.json, and have video frames in output/img, you want to train on your own data and save it to csv, you can run:
```
cd run
python train_your_own.py output/img output/labeled.json --save
```
Then your model will be saved in path pretrained_tools/pretrained_xgboost_cus.pkl, your X_train corresponding to this model will be saved in path pretrained_tools/X_train_cus.pkl, and a csv data_features.csv that contains the features of objects in your video will be stored under run/. Please feel free to adjust your model or your X_trian's locations after they have been generated.


## Authors 🔥
- **[Sunny Yang](https://github.com/crimsonsunny22)**
- **[Pu Wang](https://github.com/Puw242)**
- **[Mike Pan](https://github.com/Leo10101010)**
- **[Erchi Zhang](https://github.com/Archertakesitez)**
