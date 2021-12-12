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

    scrape_images(search_path=args.p, out_path=args.t, max_n_downloads=args.n, keyword=args.k, duplicate=duplicate)


def scroll_to_end(driver, sleep_between_interactions: int = 5):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(sleep_between_interactions)


def download_image(url: str, folder_name: str, num: int, keyword: str, duplicate: list):
    """
    Download the scraped image given an url, the folder and the image number
    :param url: url of the image
    :type url: str
    :param folder_name: folder where to store image
    :type folder_name: str
    :param num: image number
    :type num: int
    :param keyword:
    :type keyword: str
    :param duplicate:
    :type duplicate: list
    """
    # Write image to file
    reponse = requests.get(url)
    if reponse.status_code == 200:
        with open(os.path.join(folder_name, keyword + "_" + str(num) + ".jpg"), 'wb') as file:
            file.write(reponse.content)
        duplicate[url] = keyword + "_" + str(num) + ".jpg"


def scrape_images(search_path: str, out_path: str, max_n_downloads: int, keyword: str, duplicate: list):
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
    :param keyword:
    :type keyword: str
    :param duplicate:
    :type duplicate: list
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

    for i in range(1, max_n_downloads + 1):
        if i % 25 == 0:
            continue
        if i % 48 == 0:
            scroll_to_end(driver)
            containers = page_soup.findAll('div', {'class': "isv-r PNCib MSM1fd BUooTd"})
            len_containers = len(containers)
            print(len_containers)

        x_path = """//*[@id="islrg"]/div[1]/div[%s]""" % (i)

        preview_image_x_path = """//*[@id="islrg"]/div[1]/div[%s]/a[1]/div[1]/img""" % (i)
        preview_image_element = driver.find_element(By.XPATH, preview_image_x_path)

        preview_image_url = preview_image_element.get_attribute("src")

        driver.find_element(By.XPATH, x_path).click()

        # Sleep a random time after clicking on the image
        time.sleep(np.random.randint(1, 5))

        # //*[@id="islrg"]/div[1]/div[16]/a[1]/div[1]/img

        # It's all about the wait
        time_started = time.time()

        while True:
            image_element = driver.find_element(By.XPATH,
                                                """//*[@id="Sva75c"]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[
                                               2]/div[1]/a/img""")
            image_url = image_element.get_attribute('src')

            if image_url != preview_image_url:
                break
            else:
                # Making a timeout if the full res image can't be loaded
                current_time = time.time()
                if current_time - time_started > 10:
                    print("Timeout! Will download a lower resolution image and move onto the next one")
                    break

        # Downloading image
        try:
            if image_url not in duplicate:
                download_image(image_url, folder_name, i, keyword, duplicate)
                save_pickle(duplicate, out_path, DUPLICATES)
                print("Downloaded element %s out of %s total. URL: %s" % (i, max_n_downloads, image_url))
            else:
                print("Duplicate element not saved URL: %s" % (image_url))

        except:
            print("Couldn't download an image %s, continuing downloading the next one" % (i))



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
