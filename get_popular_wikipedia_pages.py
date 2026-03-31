import requests
import json
from bs4 import BeautifulSoup


def fetch_wikipedia_page(title):
    url = f"https://en.wikipedia.org/w/api.php"
    params = {
        'action': 'query',
        'format': 'json',
        'titles': title,
        'prop': 'extracts',
        'exintro': True,
        'explaintext': True,
    }
    response = requests.get(url, params=params)
    data = response.json()
    page = next(iter(data['query']['pages'].values()))
    return page['extract']


def fetch_wikipedia_page_by_url(url):
    title = url.split('/')[-1]
    return fetch_wikipedia_page(title)


def main():
    # URL of the Wikipedia Popular Pages
    url = "https://en.wikipedia.org/wiki/Wikipedia:Popular_pages"
    # Fetch the content of the page
    response = requests.get(url)
    # Parse the page with BeautifulSoup
    soup = BeautifulSoup(response.content, "html.parser")

    # Find the table containing the popular pages (first is the one we want)
    table = soup.find("table", {"class": "wikitable"})

    # Extract the links to the popular pages
    popular_pages = []
    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if cells:
            rank = cells[0].text.strip()
            link = cells[1].find("a")["href"]
            title = cells[1].find("a").text
            if rank.isnumeric():
                popular_pages.append((rank, title, "https://en.wikipedia.org" + link))

    # Print the rank, title, and link of the popular pages
    for index, (rank, title, link) in enumerate(popular_pages, start=1):
        print(f"{index}.", end="     ")
        print(f"(Rank {rank}) - {title}: {link}")
    print()

    dataset = []
    for index, (rank, title, link) in enumerate(popular_pages, start=1):
        print(f"Fetching content for")
        print(f"{index}.")
        print(f"(Rank {rank}) - {title}: {link}")

        content = fetch_wikipedia_page_by_url(link)
        print(f"Content: {content[:100]}.....\n")

        if content:
            dataset.append(
                {
                    "rank": rank,
                    "title": title,
                    "link": link,
                    "content": content,
                }
            )
        else:
            print(f"{title}: content was empty. -> not added to the dataset.\n")  # No empty content was found

    print(f"Number of pages fetched: {len(dataset)}")

    # Save the dataset to a JSON file
    with open("wikipedia_popular_pages_dataset.json", "w") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()

