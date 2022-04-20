# Closing The Gap

## Objectives

### Current Objectives

This project uses computer vision to identify where buildings are located in overhead photographs. The user can supply an image, coordinates, an address, or a csv file containing many coordinates or addresses. If the user does not supply an image, an image will be fetched using Google Static Maps API. Once images are present, each will be processed using an YOLO v5 object recognition AI model, and the end result is displayed to the end user.

### End Goal

This project will use AI (computer vision) models to identify building types from photos and maps to improve public services management efficiency such as an natural disaster evacuation plan.

## Usage

### Running the Web App

In the `prototype` directory, run:

```bash
streamlit run main.py
```
