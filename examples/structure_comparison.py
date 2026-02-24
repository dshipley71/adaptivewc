"""
Compares the structural similarity of articles across news websites.

The script:
1. Scrapes top-level navigation links from each provided domain to infer news categories.
2. Extracts at least one representative article URL per category using URL heuristics.
3. Maps site-specific categories to a predefined set of generic categories.
4. Analyzes the HTML structure of each selected article.
5. Compares article structures pairwise across domains using a change detection system.
6. Aggregates similarity scores and visualizes them as a confusion matrix.

The overall goal is to evaluate how similarly different news websites structure
content across comparable content categories.
"""


import asyncio
import json
import re
from pathlib import Path
from pprint import pprint
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement

from examples.news_monitor_ml import MLNewsMonitor, MLMonitorConfig
from crawler.adaptive.change_detector import ChangeDetector, ChangeAnalysis


COMMON_EXCLUDES = {
    "video", "videos", "live", "watch", "listen",
    "account", "login", "signup", "subscribe",
    "search", "about", "contact", "privacy",
    "terms", "jobs", "careers", "press",
    "newsletters", "weather"
}


with open("training_data/generic_categories.json", "r") as f:
  generic_categories = json.load(f)



def extract_news_categories(domains, monitor):
    """
    Extract news categories and their URLs from top-level navigation links.
    Uses MLNewsMonitor.fetch_page instead of requests.
    """
    results = {}

    for domain in tqdm(domains):
        base_url = f"https://{domain}"
        categories = {}

        try:
          html = get_html(base_url, monitor)
          if not html:
            print(f"====> {base_url} has no html")
            continue
           
        except Exception:
            results[domain] = {}
            continue

        soup = BeautifulSoup(html, "html.parser")

        nav_containers = soup.find_all(["nav", "header"])

        for container in nav_containers:
            for a in container.find_all("a", href=True):
                href = a["href"].strip()
                label = a.get_text(strip=True).lower()

                if not label or len(label) < 3:
                    continue

                full_url = urljoin(base_url, href)
                parsed = urlparse(full_url)

                if domain not in parsed.netloc:
                    continue

                path = parsed.path.strip("/")

                if not path or "/" in path:
                    continue
                if not path.isalpha():
                    continue
                if path in COMMON_EXCLUDES:
                    continue
                if len(path) > 20:
                    continue

                categories.setdefault(path, full_url)

        results[domain] = dict(sorted(categories.items()))

    if all(isinstance(v, dict) and all(isinstance(inner_v, list) and len(inner_v) == 0 for inner_v in v.values()) for v in results.values()):
      raise Exception("No news categories")
    return results


ARTICLE_EXCLUDES = {
    "video", "live", "gallery", "photos", "interactive", "index.html", "/rss"
}


def extract_category_articles(category_map, monitor, max_articles=5):
    """
    Given a category map like:
    {
        "cnn.com": {"business": "https://www.cnn.com/business"}
    }

    Return structured articles, using MLNewsMonitor.fetch_page instead of requests.
    """
    results = {}

    for domain, categories in tqdm(category_map.items()):
        print(f"Starting {domain}")
        domain_results = {}

        for category, category_url in categories.items():
            candidate_articles = []

            try:
              html = get_html(category_url, monitor)
              if not html:
                continue
            except Exception as e:
              print(f"========> Issue with getting {domain} {category} article: {e}")
              continue

            soup = BeautifulSoup(html, "html.parser")

            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                full_url = urljoin(category_url, href)
                parsed = urlparse(full_url)

                if domain not in parsed.netloc:
                    continue

                path = parsed.path.strip("/")
                segments = path.split("/")

                if len(segments) < 2:
                    continue

                article_like = (
                    re.search(r"\d{4}/\d{2}/\d{2}", path) or
                    re.search(r"\d{5,}", path) or
                    path.count("-") >= 2
                )

                if not article_like:
                    continue

                if re.search(r"/page/\d+/?$", parsed.path):
                    continue

                if not path.startswith(category):
                    continue

                candidate_articles.append((full_url, path.count("-")))

            candidate_articles = list(set(candidate_articles))
            candidate_articles.sort(key=lambda x: x[1], reverse=True)
            articles = [url for url, _ in candidate_articles[:max_articles]]

            if articles:
                domain_results[category] = {category_url: articles}

        if domain_results:
            results[domain] = domain_results

    return results

def restructure_articles(article_examples):
    article_map = []
    for domain in article_examples:
      for category in article_examples[domain]:
          for cat_url in article_examples[domain][category]:
            article_map.append({
                'domain': domain,
                'category': category,
                'url': article_examples[domain][category][cat_url][0]
            })
    return article_map

def get_generic_categories(article_examples, generic_categories_json:str):
    with open(generic_categories_json, "r") as f:
        generic_categories = json.load(f)

    article_map = []

    for domain in article_examples:
        for category in article_examples[domain]:
            generic_category = generic_categories.get(category, None)
            if generic_category:
                for cat_url in article_examples[domain][category]:
                    article_map.append({'domain': domain, 'category': generic_category, 'url': article_examples[domain][category][cat_url][0]})
    return article_map


def restructure_articles(article_examples):
    article_map = []
    for domain in article_examples:
      for category in article_examples[domain]:
          for cat_url in article_examples[domain][category]:
            article_map.append({
                'domain': domain,
                'category': category,
                'url': article_examples[domain][category][cat_url][0]
            })
    return article_map


def get_html(url, monitor):
    response = requests.get(url)
    html = response.text
    
    if response.status_code != 200:
        if html and monitor.html_captcha_check(html, url):
            print("==========> Captcha detected")
        else:
            print(f"==========> Non-200 status: {response.status_code} for {url}")
        return None

    return html

def get_article_structure(url, page_type, monitor):
  html = get_html(url, monitor)
  if not html:
    return None
  structure = monitor.structure_analyzer.analyze(html, url, page_type)
  return structure


def compare_structures(article_map, monitor):
    change_detector = ChangeDetector(logger=None)
    structure_dict = {}

    for article in tqdm(article_map):
        try:
          struct = get_article_structure(article['url'], article['category'], monitor)
          if struct is None:
            continue
          structure_dict[f"{article['domain']} {article['category']}"] = struct
        except Exception as e:
          print(f"========> Error when trying to get structure for {article['url']} {article['category']} --> {e}")
        pass

    structure_combinations = list(combinations_with_replacement(structure_dict,2))

    structure_comparisons = []

    for combo in structure_combinations:
        change_analysis = change_detector.detect_changes(
                        structure_dict[combo[0]], structure_dict[combo[1]]
                    )
        comparison_results = [combo[0], combo[1], change_analysis.similarity_score]
        structure_comparisons.append(comparison_results)
    return structure_comparisons


def confusion_matrix(data, categories_substr=None, size=(8,6)):
    # Build the set of categories from the data
    all_categories = sorted({item[0] for item in data} | {item[1] for item in data})

    # If a list of substrings is provided, filter categories that contain any of them
    if categories_substr is not None:
        categories = [
            cat for cat in all_categories
            if any(sub in cat for sub in categories_substr)
        ]
    else:
        categories = all_categories

    cat_index = {cat: i for i, cat in enumerate(categories)}
    matrix = np.zeros((len(categories), len(categories)))

    for true_cat, pred_cat, score in data:
        if true_cat in cat_index and pred_cat in cat_index:
            i = cat_index[true_cat]
            j = cat_index[pred_cat]
            matrix[i, j] += score

    fig, ax = plt.subplots(figsize=size)
    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_yticklabels(categories)

    # Add grid lines between cells
    ax.set_xticks(np.arange(-0.5, len(categories), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(categories), 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_title("Confusion Matrix")
    fig.colorbar(im)

    plt.tight_layout()
    plt.show()

    return matrix



def compare_websites(
    urls: list, 
    monitor: MLNewsMonitor, 
    generic_categories_json: str ="training_data/generic_categories.json"
):
  if not monitor:
    config = MLMonitorConfig(model_dir = "ml_models_news")
    monitor = MLNewsMonitor(config)
    monitor.start()
  news_categories = extract_news_categories(urls, monitor=monitor)
  article_examples = extract_category_articles(news_categories, monitor=monitor, max_articles=1)
  print(f"====> article examples: {article_examples}")
  article_examples = restructure_articles(article_examples)
  print(f"=====> restructured: {article_examples}")
  output = compare_structures(article_examples, monitor)
  return output


