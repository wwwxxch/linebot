from bs4 import BeautifulSoup as bs
import requests
import pandas as pd
import time

# import random
from fake_useragent import UserAgent


ua = UserAgent()
delayTimes = [0.5, 1, 1.5, 2, 2.5, 3, 3.5]


def get_ptt_post(soup):
    articles = soup.select("div.r-ent")
    result = []
    for article in articles:
        try:
            title = article.select("div.title")[0].text.strip()

            if title.startswith("[情報/知識]"):
                date = article.select("div.date")[0].text.strip()
                author = article.select("div.author")[0].text.strip()
                link = "https://www.ptt.cc" + article.select("div.title a")[0]["href"]

                # get the article
                headers = {"User-Agent": ua.random}
                res_content = requests.get(link, headers=headers)
                soup_content = bs(res_content.text, "lxml")

                # validation
                results_content = soup_content.select("span.article-meta-value")
                if len(results_content) > 3:
                    content = soup_content.find(id="main-content").text
                    footer = "※ 發信站: 批踢踢實業坊(ptt.cc),"
                    # remove texts after footer
                    content = content.split(footer)
                    main_content = content[0]
                    print(f"Done for '{title}' ")
                    pass
                else:
                    # skip if ID/版標/標題/日期為空
                    print(link, "內文異常:ID/版標/標題/日期為空")
                    print(title)
                    print(results_content)
                    print("************")
                    continue

                result.append(
                    {"title": title, "date": date, "author": author, "link": link, "content": main_content}
                )
        except Exception as e:
            print(f"Error: {e}")

        # delay = random.choice(delayTimes)
        time.sleep(0.5)

    return result


output = []
pageNumber = 5

while pageNumber > 0:
    articleListUrl = (
        f"https://www.ptt.cc/bbs/cat/search?page={pageNumber}&q=%E6%83%85%E5%A0%B1%2F%E7%9F%A5%E8%AD%98"
    )
    headers = {"User-Agent": ua.random}
    res = requests.get(articleListUrl, headers=headers)

    if res.status_code != 200:
        print(f"Failed to get page {pageNumber}")
        print(f"Error: {res.status_code}")
        print("-----------------------------")
        break

    soup = bs(res.text, "lxml")
    result = get_ptt_post(soup)
    output.extend(result)

    print(f"Done for page {pageNumber}")
    print("-----------------------------")

    pageNumber -= 1

    # delay = random.choice(delayTimes)
    time.sleep(0.5)

df = pd.DataFrame(output)
df.to_csv("crawler_docs/ptt_cat.csv")
df.head()
