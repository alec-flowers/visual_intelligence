import os
import time
import argparse

import bs4
import numpy as np
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

from utils import save_pickle, load_pickle

# SOURCE AND HOW TO INSTALL DEPENDENCIES ON LINUX
# Source Read: https://ladvien.com/scraping-internet-for-magic-symbols/
# Source YouTube: https://www.youtube.com/watch?v=Yt6Gay8nuy0

# Steps for installation:
#     Install Google Chrome (skip if its already installed)
#     Identify your Chrome version. Typically found by clicking About Google Chrome. I currently have version 77.0.3865.90 (my main version is thus 77, the number before the first dot).
#     Download the corresponding ChromeDriver from here for your main version and put the executable into an accessible location (I use Desktop/Scraping)
#     Install the Python Selenium package via pip install selenium
#
DUPLICATES = "duplicates.pickle"

def main(args):
    print("Scraping Images")
    if os.path.exists(os.path.join(args.t, DUPLICATES)):
        duplicate = load_pickle(args.t, DUPLICATES)
    else:
        print("duplicates.pickle not found, creating duplicates dict")
        duplicate = {}

    duplicate = scrape_images(search_path=args.p, out_path=args.t, max_n_downloads=args.n, keyword=args.k, duplicate=duplicate)
    save_pickle(duplicate, args.t, DUPLICATES)
    print("Duplicate Pickle Saved")


def download_image(url, folder_name, num, keyword, duplicate):
    """
    Download the scraped image given an url, the folder and the image number
    :param url: url of the image
    :type url: str
    :param folder_name: folder where to store image
    :type folder_name: str
    :param num: image number
    :type num: int
    """
    # Write image to file
    reponse = requests.get(url)
    if reponse.status_code == 200:
        with open(os.path.join(folder_name, keyword + "_" + str(num) + ".jpg"), 'wb') as file:
            file.write(reponse.content)
        duplicate[url] = keyword + "_" + str(num) + ".jpg"




def scrape_images(search_path, out_path, max_n_downloads, keyword, duplicate):
    """
    Function that scrapes images from Google (using Chrome browser).
    You have to specify the search path (enter query in google, navigate to images, copy-paste the url).
    You also have to specify the output path and how to call the directory you store the images in.
    Every time you re-execute the function, the images get overwritten, starting from 1 up to
    min(max_n_to_download, len(containers)).
    As far as I understand, we scrape as many imgs as visible on the first page,
    without loading more images via scrolling down. (TODO?)
    :param search_path: url of entered query
    :type search_path: str
    :param out_path: where to create a folder and store the images
    :type out_path: str
    :param dir_name: how to call the folder where images will be stored
    :type dir_name: str
    :param max_n_downloads: maximum number of images we want to scrape
    :type max_n_downloads: int
    """
    # Create directory for images
    folder_name = out_path
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)

    # Open the driver
    s = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=s)
    driver.get(search_path)
    driver.execute_script("window.scrollTo(0, 0);")  # Scrolling all the way up
    page_html = driver.page_source
    page_soup = bs4.BeautifulSoup(page_html, 'html.parser')
    containers = page_soup.findAll('div', {'class': "isv-r PNCib MSM1fd BUooTd"})

    len_containers = len(containers)
    # //*[@id="islrg"]/div[1]/div[1]/a[1]/div[1]

    # Determine number of images to download
    n_images = min(len_containers, max_n_downloads)

    for i in range(1, n_images + 1):
        if i % 25 == 0:
            continue

        x_path = """//*[@id="islrg"]/div[1]/div[%s]""" % (i)

        preview_image_x_path = """//*[@id="islrg"]/div[1]/div[%s]/a[1]/div[1]/img""" % (i)
        preview_image_element = driver.find_element(By.XPATH, preview_image_x_path)

        preview_image_url = preview_image_element.get_attribute("src")
        # print("preview URL", preview_image_url)
        # print(x_path)

        driver.find_element(By.XPATH, x_path).click()

        # Sleep a random time after clicking on the image
        time.sleep(np.random.randint(5, 10))

        # //*[@id="islrg"]/div[1]/div[16]/a[1]/div[1]/img

        # page = driver.page_source
        # soup = bs4.BeautifulSoup(page, 'html.parser')
        # ImgTags = soup.findAll('img', {'class': 'n3VNCb', 'jsname': 'HiaYvf', 'data-noaft': '1'})
        # print("number of the ROI tags", len(ImgTags))
        # link = ImgTags[1].get('src')
        # #print(len(ImgTags))
        # #print(link)
        #
        # n=0
        # for tag in ImgTags:
        #     print(n, tag)
        #     n+=1
        # print(len(ImgTags))

        # /html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[2]/div[1]/a/img

        # It's all about the wait
        time_started = time.time()

        while True:

            image_element = driver.find_element(By.XPATH,
                                                """//*[@id="Sva75c"]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[
                                               2]/div[1]/a/img""")
            image_url = image_element.get_attribute('src')

            if image_url != preview_image_url:
                # print("actual URL", image_url)
                break
            else:
                # making a timeout if the full res image can't be loaded
                current_time = time.time()

                if current_time - time_started > 10:
                    print("Timeout! Will download a lower resolution image and move onto the next one")
                    break

        # Downloading image
        try:
            if image_url not in duplicate:
                download_image(image_url, folder_name, i, keyword, duplicate)
                print("Downloaded element %s out of %s total. URL: %s" % (i, n_images, image_url))
            else:
                print("Duplicate element not saved URL: %s" % (image_url))
        except:
            print("Couldn't download an image %s, continuing downloading the next one" % (i))

    return duplicate

def parse_args():
    parser = argparse.ArgumentParser(description='Data labeling tool.')
    parser.add_argument("-p", type=str,
                        required=True, help="Web search path to look for images")
    parser.add_argument("-k", type=str,
                        required=True, help="Web Search Keyword used to look up images.")
    parser.add_argument("-t", type=str,
                        required=True, help="Directory that will download the images too.")
    parser.add_argument("-n", type=int,
                        required=True, help="Number of photos to scrape.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # args = argparse.Namespace(s="Good Yoga/Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_", t="test")
    main(args)
# Query: yoga warrior two
# "https://www.google.com/search?q=yoga%20warrior%20two%20-one%20-reverse&tbm=isch&tbs=rimg:CS26H6M0OM6hYWCmsFeo2fqDsgIGCgIIABAA&hl=en-US&sa=X&ved=0CBsQuIIBahcKEwj4zvXZzezzAhUAAAAAHQAAAAAQBg&biw=859&bih=847"
# PATH_W_2 = "https://www.google.com/search?q=yoga+warrior+two+&tbm=isch&ved=2ahUKEwiO1YrLzuzzAhUNixoKHbQKCf0Q2" \
#            "-cCegQIABAA&oq=yoga+warrior+two+&gs_lcp" \
#            "=CgNpbWcQAzIFCAAQgAQyBggAEAgQHjIGCAAQCBAeUJumbFibpmxg_6dsaABwAHgAgAFHiAGlAZIBATOYAQCgAQGqAQtnd3Mtd2l6LWltZ8ABAQ&sclient=img&ei=YlV6YY6cMI2WarSVpOgP&bih=847&biw=859&hl=en-US "
# dir_W_2 = 'warrior_two'
#
# # Query: yoga warrior one
# PATH_W_1 = "https://www.google.com/search?q=yoga+warrior+one&tbm=isch&ved=2ahUKEwjJ2d-Z1ezzAhWygM4BHQgoAaIQ2" \
#            "-cCegQIABAA&oq=yoga+warrior+one&gs_lcp" \
#            "=CgNpbWcQAzIFCAAQgAQyBggAEAgQHjIGCAAQCBAeMgYIABAIEB4yBAgAEBhQmL8FWI_BBWDAwgVoAHAAeACAAUmIAbwBkgEBM5gBAKABAaoBC2d3cy13aXotaW1nwAEB&sclient=img&ei=Ulx6YcnoFrKBur4PiNCEkAo&bih=847&biw=859&hl=en-US "
# dir_W_1 = 'warrior_one'
#
# # Query: yoga downward dog
# PATH_D_D = "https://www.google.com/search?q=yoga+downward+dog&tbm=isch&ved=2ahUKEwjK48DF1ezzAhUigHMKHUygAhYQ2" \
#            "-cCegQIABAA&oq=yoga+downward+dog&gs_lcp" \
#            "=CgNpbWcQAzIFCAAQgAQyBQgAEIAEMgUIABCABDIGCAAQBRAeMgYIABAFEB4yBggAEAUQHjIGCAAQCBAeMgYIABAIEB4yBggAEAgQHjIGCAAQCBAeOgQIABAYOgQIABAeUJjQBVjM2wVgueAFaABwAHgAgAFEiAHvBZIBAjEzmAEAoAEBqgELZ3dzLXdpei1pbWfAAQE&sclient=img&ei=rlx6YcrUCKKAzgPMwIqwAQ&bih=847&biw=859&hl=en-US "
# dir_D_D = 'downward_dog'

# # Scrape images
# # Warrior 1
# scrape_images(search_path=PATH_W_1, out_path=OUTPUT_PATH, dir_name=dir_W_1, max_n_downloads=max_n_to_download)
# # Warrior 2
# scrape_images(search_path=PATH_W_2, out_path=OUTPUT_PATH, dir_name=dir_W_2, max_n_downloads=max_n_to_download)
# # Downward dog
# scrape_images(search_path=PATH_D_D, out_path=OUTPUT_PATH, dir_name=dir_D_D, max_n_downloads=max_n_to_download)
