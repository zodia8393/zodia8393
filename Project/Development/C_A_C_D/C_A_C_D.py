from selenium import webdriver
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
chrome_options.add_argument("--disable-extensions")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-dev-shm-usage")

browser = webdriver.Chrome(options=chrome_options)
browser.get("chrome://settings/clearBrowserData")

cache_xpath = '//*[@id="clearBrowsingDataConfirm"]/div/div/div[2]/div[1]/div[1]/label'
cache_checkbox = browser.find_element_by_xpath(cache_xpath)
cache_checkbox.click()

button_xpath = '//*[@class="md-button md-button--secondary settings-clear-data-dialog-confirm-button"]'
button = browser.find_element_by_xpath(button_xpath)
button.click()

browser.close()
