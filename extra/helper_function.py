import zipfile as zipfile
def unzip_data(filename):
    """
    Unzips filename into the current working directory.
    Args:
    filename (str): a filepath to a target zip folder to be unzipped.
    """
    zip_ref = zipfile.ZipFile(filename, "r")
    zip_ref.extractall()
    zip_ref.close()



