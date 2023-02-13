#Youtube_Video_Download_Converter

import os
import re
import time
import requests
from selenium import webdriver
from moviepy.editor import *
import tkinter as tk

# list of search terms
search_terms = ["search term 1", "search term 2", "search term 3"]

# initialize a Chrome browser
browser = webdriver.Chrome()
browser.get("https://www.youtube.com/")

# define the function to download and convert the videos
def download_convert_videos():
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

# add a button to download and convert the videos
button = tk.Button(root, text="Download and Convert Videos", command=download_convert_videos)
button.pack()

# run the GUI
root.mainloop()
