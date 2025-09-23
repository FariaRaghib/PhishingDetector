#webscrapping part
#scrapped html files from the kaggle

import requests as re
from bs4 import BeautifulSoup
import os
#from bs4 import BeautifulSoup
URL ="https://www.kaggle.com"
response = re.get(URL)

print ("response --->" ,response,"\ntype ---->",type(response))
print ("response --->" ,response.text,"\ncontent ---->",response.content,"\nstatus_code",response.status_code)

soup=BeautifulSoup(response.text,"html.parser")

print("title with tags ----> ",soup.title, "\ntitle without tags ----> ",soup.title.text)

for link in soup.find_all('link'):
    print(link.get('href'))

print(soup.get_text())
#see txt file
#Step 1
folder="mini_dataset"
if not os.path.exists(folder):
    os.mkdir(folder)

#Step 2
def scrape_content(URL):
    response = re.get(URL)
    if response.status_code == 200:
        print("HTTP connection is successful for the URL:",URL)
        return response
    else:
        print("HTTP connection is failed for the URL:",URL)
        return None
#Step 3
path=os.getcwd() + "/" + folder
if not os.path.exists(path):
    os.mkdir(path)

def save_html(to_where,text,name):
    file_name=name+".html"
    with open(os.path.join(to_where,file_name),"w",encoding="utf-8") as f:
        f.write(text)

test_text=response.text
save_html(path,test_text,"example")

#Step 4

URL_list = [
    "https://www.kaggle.com",
    "https://www.stackoverflow.com",
    "https://quotes.toscrape.com",
    "https://www.python.org",
    "https://www.w3schools.com",
    "https://wwwen.uni.lu",
    "https://github.com",
    "https://scholar.google.com",
    "https://www.mendeley.com",
    "https://www.overleaf.com"
]

#Step 5
def create_mini_dataset(to_where,URL_list):
    for i in range(0,len(URL_list)):
        content=scrape_content(URL_list[i])
        if content is not None:
            save_html(to_where,content.text,str(i))
        else:
            pass
    print("Mini dataset is created!")

create_mini_dataset(path,URL_list)
