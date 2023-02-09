from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import datetime

# Create a new instance of the Chrome driver
driver = webdriver.Chrome()

# Navigate to the Naver login page
driver.get("https://nid.naver.com/nidlogin.login")

# Find the username and password input fields
username = driver.find_element_by_name("id")
password = driver.find_element_by_name("pw")

# Enter your login credentials
username.send_keys("your_email_address")
password.send_keys("your_password")

# Submit the login form
driver.find_element_by_xpath('//*[@id="frmNIDLogin"]/fieldset/input').click()

# Wait for the page to load
time.sleep(5)

# Navigate to the Naver email page
driver.get("https://mail.naver.com")

# Wait for the email list to load
WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="container"]/div[2]/div[1]/div[1]/div[2]/div[2]/table/tbody')))

# Find the email table
email_table = driver.find_element_by_xpath('//*[@id="container"]/div[2]/div[1]/div[1]/div[2]/div[2]/table/tbody')

# Get the list of email rows
email_rows = email_table.find_elements_by_tag_name("tr")

# Iterate over the email rows
for email_row in email_rows:
    # Get the date element
    date_element = email_row.find_element_by_xpath(".//td[@class='date']")

    # Get the date string
    date_str = date_element.get_attribute("title")

    # Parse the date string
    email_date = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M")

    # Check if the email is not from today
    if email_date.date() != datetime.datetime.today().date():
        # Find the delete button
        delete_button = email_row.find_element_by_xpath(".//span[@class='btn_delete']")

        # Click the delete button
        delete_button.click()

# Close the browser window
driver.quit()
