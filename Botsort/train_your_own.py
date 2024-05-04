import json
import sys
def labled_json_to_csv(arg_json:str)->None:
    """
    This function takes a json file as the labled boxes and returns a csv file
    that contains all the features required

    Args:
        arg_json: the json file's path. This json file contains user's labled boxes within a video
    """
    pass

if __name__ == "__main__":
    json_path = sys.argv[1] #first argument is the labled boxes json file
    labled_json_to_csv(arg_json=json_path)