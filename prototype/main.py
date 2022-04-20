import streamlit as st
import pandas as pd
from map_scraper import map_api
from map_scraper import map_api_df
import os

def classify_image(image):
    try:
        file_name = image.name
        with open(f"temp/input/{file_name}","wb") as f:
            f.write(image.getbuffer())
    except:
        file_name = 'file.png'
        with open(f"temp/input/{file_name}","wb") as f:
            f.write(image)

    with st.spinner("Finding Buildings..."):
        os.system('python3 yolov5/detect.py --weights best.pt --img 320 --conf 0.35 --source "temp/input" --project "temp/output" --exist-ok')
    st.image(f'temp/output/exp/{file_name}')
    os.remove(f'temp/input/{file_name}')
    os.remove(f'temp/output/exp/{file_name}')

def webapp():
    # Selection Menu for what the user wants to upload
    option = st.selectbox("Select Mode", ['Image', 'Geocode', 'Address', 'Bulk Geocode CSV', 'Bulk Address CSV'])

    # Image Selection
    if option == 'Image':
        image = st.file_uploader("Upload Image", type=['png'])
        if image is not None and st.button("Run"):
            classify_image(image)

    # Geocode Selection
    elif option == 'Geocode':
        lat = st.text_input("Latitude")
        lon = st.text_input("Longitude")

        # Check that the inputs are numbers
        valid_numbers = False
        try:
            lat = float(lat)
            lon = float(lon)
            st.markdown(f"Coordinates: ({lat}, {lon})")
            valid_numbers = True
        except:
            st.markdown("Please enter a number")
        if valid_numbers and st.button("Run"):
            image = map_api(lat, lon, 20, 'google')
            classify_image(image)

    # Address Selection
    elif option == 'Address':
        address = st.text_input("Address")
        st.markdown(f"Address: {address}")
        # TODO

    # CSV Selection
    elif option == 'Bulk Geocode CSV' or 'Bulk Address CSV':
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded_file is not None and st.button("Run"):
            df = pd.read_csv(uploaded_file)
            st.write(df)

            if option == 'Bulk Geocode CSV':
                with st.spinner("Requesting Images"):
                    image_list = map_api_df(df, 20, 'google')
                for image in image_list:
                    st.markdown(image['name'])
                    classify_image(image['image'])

            if option == 'Bulk Address CSV':
                return
                # TODO


    # Catch case if none of the above states are triggered
    # Should not be reachable
    else:
        st.markdown("No option selected.")


webapp()