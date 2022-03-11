# Import needed libraries
from PIL import Image
from itsdangerous import exc
from config.credentials import GOOGLE_API_KEY
from tqdm import trange
from io import BytesIO
import pandas as pd
import requests
import os


def map_api(filename: str, zoom: int, map_api: str, num_images: int):
    """
    Summary:
        This image scraper is used to build the training and test data for any CNN model by extracting images from google or mapbox images. 

    Args:
        filename (str): name of the directory where cooridantes are located
        zoom (int): The zoom level of the map
        map_api (str): Determine which api to use between google='google' and mapbox='mapbox'
        num_images (int): The number of images that will be returned
    """
    
    # Check the name of file
    file_name = os.path.basename(filename)
    sub_dir = os.path.splitext(file_name)[0]
   
    # Iterate through csv files and convert to pandas format
    df = pd.read_csv(filename)

    # create a counter for num of images
    counter = 0

    while counter < num_images:

        for idx in trange(len(df), desc="Processing images"):
            latitude = df['latitude'][idx]
            longitude = df['longitude'][idx]
            company_name = df['name'][idx]
            sub_category = df['FeatureSubType'][idx]
            # print(longitude, latitude, company_name)

            # Create directory if it doesn't exist
            isExist = os.path.exists('/Users/nathanieljacques/Rel8ed.to/CTG/prototype/data/img/' + sub_category)

            if not isExist:
                os.makedirs('/Users/nathanieljacques/Rel8ed.to/CTG/prototype/data/img/' + sub_category)

            if map_api == 'google':
                    # Connect to API server
                    res_google = requests.get(f'https://maps.googleapis.com/maps/api/staticmap?center={latitude},{longitude}&zoom={zoom}&size=300x300&maptype=satellite&key={GOOGLE_API_KEY}')


                    with open(f"{company_name}.png", "wb") as f:
                        f.write(res_google.content)
                        try:
                            image = Image.open(f"{company_name}.png")
                            image.save(f'/Users/nathanieljacques/Rel8ed.to/CTG/prototype/data/img/{sub_category}/{company_name + ".png"}', 'png')
                            # image.show(f"{company_name}.png")
                        except FileNotFoundError:
                            image.save(f'/Users/nathanieljacques/Rel8ed.to/CTG/prototype/data/reject/{company_name + ".png"}', 'png')
                        except OSError:
                            image.save(f'/Users/nathanieljacques/Rel8ed.to/CTG/prototype/data/reject/{company_name + ".png"}', 'png')

                    counter += 1

            if counter == num_images:
                print("images Proccessing completed....")
                break 

        
# if __name__ == '__main__':
    # neo4j_emergency = map_api('/Users/nathanieljacques/Rel8ed.to/CTG/prototype/data/neo4j_emergency.csv',18, 'google', num_images=100)
    # transportation = map_api('/Users/nathanieljacques/Rel8ed.to/CTG/prototype/data/Transportation.csv', 18, 'google', num_images=350)
    # emergency = map_api('/Users/nathanieljacques/Rel8ed.to/CTG/prototype/data/CentersOfWorship.csv', zoom=18, map_api='google', num_images=350)
    
