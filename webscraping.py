import requests
from bs4 import BeautifulSoup

def scrape_resources(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    resources = []
    for resource_div in soup.find_all('div', class_='resource'):
        title = resource_div.find('h3').text
        link = resource_div.find('a')['href']
        topic = resource_div['data-topic']
        resources.append({
            'topic': topic,
            'title': title,
            'url': link
        })

    return resources

url = "www.palabrar.com" 
resources = scrape_resources(url)