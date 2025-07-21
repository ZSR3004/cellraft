import os
import gc 
import psutil

def get_paths(path: str, filename_ext: str = ".npy", alphabetize: bool=True) -> list[str]:
    """
    Get all file paths in a directory and its subdirectories.
    
    Args:
        path (str): The root directory to search for files.
        filename_ext (str): The file extension to filter by (default is ".npy").
        alphabetize (bool): Whether to sort the paths alphabetically (default is True).

    Returns:
        list[str]: A list of file paths that match the specified extension.
    """
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(filename_ext):
                    paths.append(os.path.join(root, file))
    if alphabetize:
        paths.sort()
    return paths
