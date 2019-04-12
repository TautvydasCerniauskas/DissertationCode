import requests
from bs4 import BeautifulSoup

def retrieve_page(url):

    get_page = requests.get(url, timeout=5)
    page_html = BeautifulSoup(get_page.content, "html.parser")

    return page_html

def get_list_of_links(page):

    list_of_links = page.findAll('a')
    return list_of_links

def get_href(list_of_links):

    list_of_href = []

    for href in list_of_links:
        if href.has_attr('href'):
            list_of_href.append(href.attrs['href'])

    return list_of_href

def create_title_and_episode_number(href, links):

    list_of_episode_number = []
    list_of_titles = []

    for link in href:
        list_of_episode_number.append(link.split('/')[1].split('.')[0])

    for episode_no, title in zip(list_of_episode_number, links):
        if title.has_attr('href'):
            list_of_titles.append(episode_no + ' ' + ' '.join(title.text.split()[1:]))

    return list_of_titles

def write_info_into_file(titles):
    with open('../data/raw/titles.txt', 'w', encoding='utf-8') as title_file:
        for title in titles:
            title_file.write(title + '\n')

if __name__ == '__main__':
    url = 'https://fangj.github.io/friends/'
    page = retrieve_page(url)
    links = get_list_of_links(page)
    href = get_href(links)
    titles = create_title_and_episode_number(href, links)
    write_info_into_file(titles)
