import requests
from bs4 import BeautifulSoup
import os
import json
import time
from tqdm import tqdm
import random

url = "https://journals.plos.org/plosntds/volume"
root_url = "https://journals.plos.org"
target_year = "2026"
target_month = "March"

#response = requests.get(url)
content = requests.get(url).content
#print("response headers:", response.headers)
#print("content:", content)


def print_name(tag):
    for child in tag.children:
        if child.name is not None:
            print_name(child)
            if child.name == "div":
                print(child.name)
                #print(child.class)
    
    
soup = BeautifulSoup(content, "html.parser")
#print(soup.name)

#print_name(soup)


slides = soup.find("ul", id="journal_slides")
target_year_slide = slides.find("li", id=target_year)
if target_year_slide is None:
    raise ValueError(f"Cannot find year {target_year} on the volume page")

target_issue_link = None
for a in target_year_slide.find_all("a"):
    month = a.find("span")
    if month and month.get_text(strip=True) == target_month:
        target_issue_link = root_url + a.get("href")
        break

if target_issue_link is None:
    raise ValueError(f"Cannot find {target_month} {target_year} on the volume page")

links = [target_issue_link]
print(f"Target issue: {target_month} {target_year} -> {target_issue_link}")


article_urls = []
print("Catching article urls...")
for link in tqdm(links):
    content = requests.get(link).content
    soup = BeautifulSoup(content, "html.parser")
    divs = soup.find_all("div", class_="section")
    
    res_div = None
    for div in divs:
        a = div.find("a")
        if a and a.get("title") == "Research Article":
            res_div = div
            break
    if res_div is None:
        print(f"No Research Article section found in {link}")
        continue
    #print(res_div)
    ps = res_div.find_all("p", class_="article-info")
    
    for p in ps:
        a = p.find_all("a")[-1]
        article_urls.append(a.get("href"))
        #print(a.get("href"))

    #time.sleep(random.uniform(0, 0.1))
    #break


i = 0
print("Catching article contents...")
for article_url in tqdm(article_urls):
        #article_url = "https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0011968"
        #article_url = "https://doi.org/10.1371/journal.pntd.0011755"
    try:
        content = requests.get(article_url).content
        soup = BeautifulSoup(content, "html.parser")
        pdf_url = root_url + soup.find("div", class_= "dload-pdf").find("a").get("href")
        
        author_summary_section = soup.find_all("div", class_="abstract toc-section abstract-type-summary")[-1]
        try:
            author_summary_section.find("h2").decompose() 
        except:
            pass
        author_summary = author_summary_section.get_text().strip()
        doi = soup.find("meta", {"name": "citation_doi"}).get("content")
        
        article_features = {}
        pubdate = soup.find("li", {"id": "artPubDate"}).get_text()
        title = soup.find("meta", {"name": "citation_title"}).get("content")
        article_features["Title"] = title
        article_features["URL"] = article_url
        article_features["Published"] = pubdate.split(":")[1].strip()
        
        year = pubdate.split(":")[1].split(",")[-1].strip()
        month = pubdate.split(" ")[1].split(" ")[0]
        #print(year, month)
        
        subject_list = soup.find("ul", {"id": "subjectList"})
        
        subject_areas = [a.get_text().strip() for a in subject_list.find_all("a", class_="taxo-term")]
        
        article_features["Subject_Areas"] = subject_areas
        
        
        #print(doi)
        folder_path = f"./PLOS_ntds/{doi.split('/')[0]}/{year}/{month}/{doi.split('/')[-1]}/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        #with open(folder_path + "author_summary.json", "w") as file:
        #    json.dump(author_summary, file, indent=4)
        with open(folder_path + "author_summary.txt", "w", encoding="utf-8") as file:
            file.write(author_summary)
        
        
        authors = soup.find_all("meta", {"name": "citation_author"})
        authors_institutions = {}
        for author in authors:
            name = author.get("content")
            institutions = []
            
            next_tag = author.find_next_sibling()
            while next_tag and next_tag.get("name") == "citation_author_institution":
                institutions.append(next_tag.get("content"))
                next_tag = next_tag.find_next_sibling()

            authors_institutions[name] = institutions
        
        #print(authors_institutions)
        article_features["Authors"] = authors_institutions
        #print(article_features)
        with open(folder_path + "article_features.json", "w", encoding="utf-8") as file:
            json.dump(article_features, file, indent=4, ensure_ascii=False)
        
            
        abstract = soup.find("div", class_="abstract toc-section abstract-type-")
        abstract_dict = {}
        sections = abstract.find_all("h3")
        #print(sections)
        if len(sections) == 0:
            para = abstract.find("p").get_text().strip()
            abstract_dict["Abstract"] = para
            #print("No subsection for abstract! " + article_url)
        else:
            for section in sections:
                para = section.find_next("p").get_text().strip()
                abstract_dict[section.get_text().strip()] = para

        #print(abstract_dict)
        
        with open(folder_path + "abstract.json", "w", encoding="utf-8") as file:
            json.dump(abstract_dict, file, indent=4, ensure_ascii=False)
        
        
        
        content_sections = soup.find_all("div", {"xmlns:plos": "http://plos.org", "class":"section toc-section"})
        content = {}
        figure = {}
        
        for content_section in content_sections:
            section_title = content_section.find("a").get("title")
            if section_title:
                if section_title.lower() == "supporting information":
                    continue
                
                content_section.find("h2").decompose()
                
                figure_tables = content_section.find_all("div", class_="figure-inline-download")
                for figure_table in figure_tables:
                    figure_name = figure_table.find_next("div", class_="figcaption")
                    figure_url = "https://journals.plos.org/plosntds/" + figure_table.find_all("a")[-1].get("href")
                    name = figure_name.get_text().strip()
                    figure[name.split(".")[0]] = [name, figure_url]
                    figure_src = figure_name.find_next("p", class_="caption_object")
                    
                    figure_name.decompose() 
                    figure_src.decompose() 
                    figure_table.decompose() 
                    
                #print(figure)
                para = content_section.get_text().strip()
                if section_title.lower() == "acknowledgments":
                    acknowledgments = para
                else:
                    content[section_title] = para
                #print(section_title)
            else:
                raise ValueError("No title for the section")
        #print(content)
        #print(figure)
        
        with open(folder_path + "acknowledgments.txt", "w", encoding="utf-8") as file:
            file.write(acknowledgments)
        
        with open(folder_path + "content.json", "w", encoding="utf-8") as file:
            json.dump(content, file, indent=4, ensure_ascii=False)
            
        '''
        ### Crawling figures, tables, and pdfs
        
        if not os.path.exists(folder_path + "src/"):
            os.makedirs(folder_path + "src/")
        for name, url in figure.items():
            response = requests.get(url[1])
            if response.status_code == 200:
                file_path = folder_path + "src/" + f"{name}.tiff"
                with open(file_path, "wb", encoding="utf-8") as f:
                    f.write(response.content)
            else:
                print(f"Failed to download {url[1]}")
                
        with open(folder_path + "figure_table.json", "w", encoding="utf-8") as file:
            json.dump(figure, file, indent=4, ensure_ascii=False)


        response = requests.get(pdf_url)
        if response.status_code == 200:
            file_path = folder_path + "src/" + f"{doi.split('/')[1]}.pdf"
            with open(file_path, "wb", encoding="utf-8") as f:
                f.write(response.content)
        else:
            print(f"Failed to download {pdf_url}")
        '''
        #break
        
        #time.sleep(random.uniform(0.1, 0.5))
        i += 1
        
        #break
        
    except KeyboardInterrupt:
        print("Program was stopped by user.")
        break
    except Exception as err:
        print("Cannot catch " + article_url)
        print("Error: ", err)
    
    
    

print(f"Dataset established successfully! Containing {i} articles!")
