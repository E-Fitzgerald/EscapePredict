import requests
from bs4 import BeautifulSoup
import re
import csv

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:55.0) Gecko/20100101 Firefox/55.0',
}
# URL = "https://www.escapeauthority.com/list/"
# page = requests.get(URL, headers=headers)
# soup = BeautifulSoup(page.content, "html.parser")
# reviews = []
# for game in soup.find_all("a"):
#     r = game.get("href")
#     if r and "review" in r:
#         reviews.append(game.get("href"))

# f = open("esc_auth.txt","w+")
# for r in reviews:
#     f.write(r + "\n")
# f.close() 

f= open("esc_auth.txt","r")
if f.mode == "r":
    reviews = f.read()
reviews = list(reviews.split("\n"))
revs = [review[:len(review)-1] for review in reviews if "https" in review][1:]

with open('esc_auth_revs.csv', 'w', newline='') as csvfile:
    fieldnames = ['company', 'game', 'score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for i in range(len(revs)):
        r = revs[i]
        try:
            ind_page = requests.get(r, headers=headers)
            ind_soup = BeautifulSoup(ind_page.content, "html.parser")

            key = 0
            for img in ind_soup.findAll('img'):
                alt = img.get('alt')
                if alt and "Keys" in alt:
                    key = int(list(alt.split(" "))[0])
                    break
            
            instructions = ind_soup.find("span", itemprop="itemReviewed")
            if instructions:
                wo = str(instructions)[30:len(instructions)-8]
                tup = wo.split(" - ")
                if len(tup)==2:
                    company = tup[0]
                    game = tup[1]
                    writer.writerow({"company":company, "game": game, "score": key})
        except:
            print("Something went wrong at:", i)
        if i%25==0:
            print("I'm on", i, "reviews!")