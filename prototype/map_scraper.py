from PIL import Image, ImageFile
from credentials import GOOGLE_API_KEY, MAP_BOX_API_KEY
from io import BytesIO
import pandas as pd
import requests
import os


def map_api(filename: str, zoom: int, map_api: str):
    """
    Summary:
        This image scraper is used to build the training and test data for any CNN model by extracting images from google or mapbox images. 

    Args:
        filename (str): name of the directory where cooridantes are located
        zoom (int): The zoom level of the map
        map_api (str): Determine which api to use between google='google' and mapbox='mapbox'
    """
    
    # Check the name of file
    file_name = os.path.basename(filename)
    sub_dir = os.path.splitext(file_name)[0]

    # Create directory if it doesn't exist
    isExist = os.path.exists('prototype/train_data/images/' + sub_dir)

    if not isExist:
        os.makedirs('prototype/train_data/images/' + sub_dir)
    
    # Iterate through csv files and convert to pandas format
    df = pd.read_csv(filename)
    print(df)
    
    for idx in range(len(df)):

        latitude = df['latitude'][idx]
        longitude = df['longitude'][idx]
        company_name = df['name'][idx]
        
        # skip image if it already exists. If you want to refresh existing images, please delete the old ones
        # first or comment out these two lines
        if os.path.exists(f"prototype/train_data/images/{sub_dir}/{company_name}.png"):
            continue

        print(longitude, latitude, company_name)

        if map_api == 'google':
            # Connect to API server
            res_google = requests.get(
                f'https://maps.googleapis.com/maps/api/staticmap?center={latitude},{longitude}&zoom={zoom}&size=600x600&maptype=satellite&key={GOOGLE_API_KEY}')
            
            # # Truncate the png files 
            # ImageFile.LOAD_TRUNCATED_IMAGES = True
            
            with open(f"prototype/train_data/images/{sub_dir}/{company_name}.png", "wb") as f:
               
                f.write(res_google.content)
    
                #image = Image.open(f"{company_name}.png")
                #image.save(f'prototype/train_data/images/{sub_dir}/{company_name + ".png"}','png')
                #image.show(f"{company_name}.png")
    
        elif map_api == 'mapbox':
            res_map_box = requests.get(
                f'https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{longitude},{latitude},{zoom}/600x600?access_token={MAP_BOX_API_KEY}')

            with open("gfg.png", "wb") as f:
                f.write(res_map_box.content)
    
                image = Image.open("gfg.png")
                image.save(f'prototype/train_data/images/{sub_dir}/{company_name + ".png"}', 'png')

        else:
            print("Please specify static map API server")


if __name__ == '__main__':
    # corp = map_api('CTG/prototype/train_data/corporatebuilding.csv', 21, 'google')
    manufact = map_api('prototype/train_data/csvs/religious.csv', 20, 'google')
    
