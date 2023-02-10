#Auto Image Download & Preprocess 

import requests
from bs4 import BeautifulSoup
from PIL import Image
import os

def download_image(url, file_path):
    response = requests.get(url)
    with open(file_path, 'wb') as f:
        f.write(response.content)

def preprocess_image(file_path, output_folder, size=(128, 128), grayscale=False):
    with Image.open(file_path) as im:
        im = im.resize(size)
        if grayscale:
            im = im.convert('L')
        output_path = os.path.join(output_folder, os.path.basename(file_path))
        im.save(output_path)

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def get_image_urls(search_term):
    search_url = f'https://www.google.com/search?q={search_term}&tbm=isch'
    response = requests.get(search_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    image_elements = soup.find_all('img')
    return [img.get('src') for img in image_elements]

# Example usage
search_term = 'dog'
input_folder = 'raw_images'
output_folder = 'processed_images'

image_urls = get_image_urls(search_term)

create_folder_if_not_exists(input_folder)
create_folder_if_not_exists(output_folder)

for i, url in enumerate(image_urls):
    file_path = os.path.join(input_folder, f'{i}.jpg')
    download_image(url, file_path)
    preprocess_image(file_path, output_folder, grayscale=True)
