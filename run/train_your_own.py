from get_feature import get_features_multi
from prepare_training import make_classifier_custom
import argparse
import os

#SUB ENTRY FILE (loading custom model)
#tested!
def labeled_json_to_model(multi_img_path: str,
    labeled_json_path: str,
    old_data_path = "",
    save=True)->None:
    """
    This function takes a json file which contains labled boxes load the custom models.

    Args:
        multi_img_path: the directory that contains images corresponding to the labeled json file
        labeled_json_path: the path to the labeled json file
        old_data_path: the old csv file that contains saved features, to which you wnat to append new
        feature rows on.
        save: whether you want to save the csv file as output. Default is true
    """
    df = get_features_multi(multi_img_path=multi_img_path, labeled_json_path=labeled_json_path,
                            old_data_path=old_data_path)
    if save:
        df.to_csv("data_feature.csv")
    
    make_classifier_custom(df=df)

#tested!
def main():
    parser = argparse.ArgumentParser(description='let\'s use your labeled data to load your custom model!')
    parser.add_argument('images_dir', help='directory containing images corresponding to the labeled json file')
    parser.add_argument('labeled_json', help = 'your labeled json file')
    parser.add_argument('--old_path', default = "", help='the csv file(optional) you want to add new rows on')
    parser.add_argument('--save', action='store_true', help='whether you want to save the output feature csv file')

    args = parser.parse_args()
    multi_img_path = args.images_dir
    labeled_json_path = args.labeled_json
    old_data_path = args.old_path
    save = False
    if args.save:
        save = True
    labeled_json_to_model(multi_img_path = multi_img_path, labeled_json_path = labeled_json_path, old_data_path = old_data_path,
                      save = save)
#tested!
if __name__ == "__main__":

    main()
    """
    multi_img_path = "labeled_data/chase_1_left_half_9520_9632/img"
    labeled_json = "labeled_data/chase_1_left_half_9520_9632/labeled.json"
    labeled_json_to_model(multi_img_path=multi_img_path, labeled_json_path=labeled_json, save=False)
    """