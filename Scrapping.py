# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 12:07:37 2022

@author: DELL
"""
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import pandas as pd

# Initialize Chrome driver using Service object
chrome_driver_path = 'D:/labs/Semester 6/chromedriver-win64/chromedriver.exe'
service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service)

# Function to scrape data from URL and append to CSV
def scrape_and_append(url):
    # Lists to store data
    products = []

    for i in range(1, 10):
        driver.get(url)
        content = driver.page_source
        soup = BeautifulSoup(content, 'html.parser')
    
        # Find product containers
        for product in soup.find_all('li'):
            # Product name
            name = product.find('div', class_='cmp-result-name')
            if name:
                names = name.text.strip()
                products.append(names)
        
        # Create DataFrame
    data = {
        'Disease Name': products,
    }
    df = pd.DataFrame(data)

    # Append to CSV
    with open('amazonfashionZ.csv', 'a', newline='', encoding='utf-8') as f:
        df.to_csv(f, header=f.tell()==0, index=False)

# List of URLs to scrape
urls = [
    "https://www.mayoclinic.org/diseases-conditions/index?letter=Z"
   
    # Add more URLs here as needed
]

# Scrape data from each URL and append to CSV
for url in urls:
    scrape_and_append(url)

# Close the WebDriver
driver.quit()

