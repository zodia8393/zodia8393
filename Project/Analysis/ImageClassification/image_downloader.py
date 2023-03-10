import os
import urllib.request
import numpy as np
import cv2


def download_and_preprocess_images(query, num_images, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    image_urls = get_image_urls(query, num_images)
    for i, image_url in enumerate(image_urls):
        try:
            img = download_and_preprocess_image(image_url)
            if img is not None:
                filename = os.path.join(save_dir, f'{i:05}.jpg')
                cv2.imwrite(filename, img)
        except Exception as e:
            print(f'Error occurred while processing image {image_url}. Reason: {e}')


def delete_downloaded_images(image_dir):
    if os.path.exists(image_dir):
        for image_file in os.listdir(image_dir):
            try:
                img_path = os.path.join(image_dir, image_file)
                os.remove(img_path)
            except Exception as e:
                print(f'Failed to delete {img_path}. Reason: {e}')
    else:
        print(f'{image_dir} does not exist')


def get_image_urls(query, num_images):
    urls = []
    query = query.replace(' ', '+')
    search_url = f'https://www.google.com/search?q={query}&tbm=isch'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
    req = urllib.request.Request(search_url, headers=headers)
    html = urllib.request.urlopen(req).read()
    html = str(html)
    results_start = html.find("class=\"rg_i Q4LuWd\"")
    for i in range(num_images):
        try:
            results_start = html.find("class=\"rg_i Q4LuWd\"", results_start + 1)
            results_end = html.find("\"></div>", results_start)
            url_raw = html[results_start + 23: results_end]
            if 'http' in url_raw:
                urls.append(url_raw)
        except Exception as e:
            print(f'Error occurred while getting image urls. Reason: {e}')
            continue
    return urls


def download_and_preprocess_image(image_url):
    req = urllib.request.urlopen(image_url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is not None:
        img = preprocess_image(img)
    return img


def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        if w > 50 and h > 50:
            img = img[y:y+h, x:x+w]
            img = cv2.resize(img, (224, 224))
            return img
    return None
