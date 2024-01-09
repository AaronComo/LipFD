import os

def get_list(path) -> list:
    r"""Recursively read all files in root path"""
    image_list = list()
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.split('.')[1] in ['png', 'jpg', 'jpeg']:
                image_list.append(os.path.join(root, f))
    return image_list