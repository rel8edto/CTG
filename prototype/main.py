import streamlit as st
import pandas as pd
from map_scraper import map_api
from map_scraper import map_api_df

def webapp():
    # Selection Menu for what the user wants to upload
    option = st.selectbox("Select Mode", ['Image', 'Geocode', 'Address', 'Bulk Geocode CSV', 'Bulk Address CSV'])

    # Image Selection
    if option == 'Image':
        image = st.file_uploader("Upload Image", type=['jpeg', 'jpg', 'png'])
        if image is not None and st.button("Run"):
            st.image(image)

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
        if valid_numbers:
            image = map_api(lat, lon, 20, 'google')
            st.image(image)

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
                    st.image(image['image'])

            if option == 'Bulk Address CSV':
                return
                # TODO


    # Catch case if none of the above states are triggered
    # Should not be reachable
    else:
        st.markdown("No option selected.")


webapp()