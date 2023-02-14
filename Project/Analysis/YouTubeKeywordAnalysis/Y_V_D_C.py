import importlib

def install_module(module_name):
    try:
        importlib.import_module(module_name)
    except ImportError:
        import subprocess
        subprocess.check_call([
            "pip", "install", module_name
        ])

required_modules = ["os", "re", "time", "requests","selenium","moviepy","tkinter"]

for module_name in required_modules:
    install_module(module_name)
    
import selenium

current_version = selenium.__version__
if LooseVersion(current_version) > LooseVersion("4.0.0"):
    !pip install selenium==4.0.0

import os
import re
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from moviepy.editor import *
import tkinter as tk


# set the options to run the browser in headless mode
options = Options()
options.add_argument('--headless')

# initialize a Chrome browser in headless mode
browser = webdriver.Chrome(options=options)
browser.get("https://www.youtube.com/")

# define the function to download and convert the videos
def download_convert_videos():
    search_terms = search_term_entry.get().split(",")

    for search_term in search_terms:
        # search for the desired video
        search_box = browser.find_element_by_xpath("//input[@id='search']")
        search_box.clear()
        search_box.send_keys(search_term)
        search_box.submit()

        # find the top 10 most viewed videos
        time.sleep(2) # wait for the search results to load
        videos = browser.find_elements_by_xpath("//a[@title='video title']")
        videos = videos[:10]

        # iterate over the top 10 most viewed videos
        for i, video in enumerate(videos):
            video.click()

            # extract the video URL
            video_url = browser.current_url
            video_id = re.search("v=(.*)", video_url).group(1)
            download_url = f"https://www.youtube.com/watch?v={video_id}"

            # download the video
            r = requests.get(download_url)
            with open(f"video_{search_term}_{i}.mp4", "wb") as f:
                f.write(r.content)

            # convert the video to audio
            video = VideoFileClip(f"video_{search_term}_{i}.mp4")
            audio = video.audio
            audio.write_audiofile(f"audio_{search_term}_{i}.wav")

            # go back to the search results page
            browser.back()

    # close the browser
    browser.close()

# create the GUI screen
root = tk.Tk()
root.title("Download and Convert Videos")

# add a label and entry to enter search terms
search_term_label = tk.Label(root, text="Search Terms (separated by commas):")
search_term_label.pack()
search_term_entry = tk.Entry(root)
search_term_entry.pack()

# add a button to download and convert the videos
button = tk.Button(root, text="Download and Convert Videos", command=download_convert_videos)
button.pack()

# run the GUI
root.mainloop()
