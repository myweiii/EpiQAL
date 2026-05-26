import os
import re
import json
import time
import random
from bs4 import BeautifulSoup
from curl_cffi import requests as creq
from tqdm import tqdm

# ----------------------------- config -----------------------------
ROOT_URL      = "https://academic.oup.com"
TARGET_YEAR   = "2025"
TARGET_MONTH  = None          # e.g. "February" / "April" / ... ; None = all issues in TARGET_YEAR
SECTION_TITLE = "Original Articles"
OUT_DIR       = "./IJE"

# Cloudflare on academic.oup.com inspects TLS/JA3. curl_cffi impersonates real
# browsers; we rotate profiles because some get throttled intermittently.
IMPERSONATES = ["chrome124", "chrome119", "chrome131", "chrome120",
                "chrome133a", "chrome116", "safari17_0"]


# ----------------------------- HTTP -------------------------------
def fetch(url, max_attempts=10, base_sleep=2.0):
    """GET with rotating browser fingerprints; retries on Cloudflare 403/challenge."""
    last = None
    for i in range(max_attempts):
        prof = IMPERSONATES[i % len(IMPERSONATES)]
        try:
            r = creq.get(
                url, impersonate=prof, timeout=30,
                headers={"Accept-Language": "en-US,en;q=0.9"},
            )
            if r.status_code == 200 and "Just a moment" not in r.text[:1000]:
                return r.text
            last = f"HTTP {r.status_code}"
        except Exception as e:
            last = str(e)
        time.sleep(random.uniform(base_sleep, base_sleep + 2))
    raise RuntimeError(f"Failed after {max_attempts} attempts: {url} ({last})")


# ----------------------------- parsing helpers --------------------
def get_issue_links(year, month=None):
    """Return list of (label, url) for IJE issues in `year`, optionally filtered by month."""
    archive_url = f"{ROOT_URL}/ije/issue-archive/{year}"
    soup = BeautifulSoup(fetch(archive_url), "html.parser")
    widget = soup.find("div", class_="widget-IssuesAndVolumeListManifest")
    if widget is None:
        raise ValueError(f"No issue-list widget at {archive_url}")
    issues = []
    for a in widget.find_all("a", href=lambda h: h and "/ije/issue/" in h):
        label = a.get_text(strip=True)        # e.g. "Volume 55, Issue 1, February"
        if month and month.lower() not in label.lower():
            continue
        issues.append((label, ROOT_URL + a.get("href")))
    return issues


def get_section_article_urls(issue_url, section_title=SECTION_TITLE):
    """Pick out every article link under the given section heading on an issue TOC page."""
    soup = BeautifulSoup(fetch(issue_url), "html.parser")
    target_h4 = None
    for h4 in soup.find_all("h4", class_="act-header"):
        if h4.get_text(strip=True).lower() == section_title.lower():
            target_h4 = h4
            break
    if target_h4 is None:
        return []
    group = target_h4.find_next_sibling("div", class_="al-article-list-group")
    if group is None:
        return []
    urls = []
    for h5 in group.find_all("h5", class_="item-title"):
        a = h5.find("a", class_="at-articleLink")
        if a and a.get("href"):
            urls.append(ROOT_URL + a["href"])
    return urls


def parse_citation_meta(soup):
    """Walk citation_* meta tags. Author institutions follow their author in document order."""
    out = {}
    keywords = []
    authors = {}
    current_author = None
    for m in soup.find_all("meta"):
        name = m.get("name") or ""
        content = m.get("content") or ""
        if not name.startswith("citation_"):
            continue
        if name == "citation_author":
            current_author = content
            authors.setdefault(current_author, [])
        elif name == "citation_author_institution":
            if current_author is not None:
                authors[current_author].append(content)
        elif name == "citation_keyword":
            keywords.append(content)
        elif name not in out:
            out[name] = content
    out["_authors"] = authors
    out["_meta_keywords"] = keywords
    return out


def parse_keywords(soup, meta_keywords):
    """Prefer <div class="kwd-group"> author keywords; fall back to citation_keyword meta."""
    kg = soup.find("div", class_="kwd-group")
    if kg:
        kws = [a.get_text(strip=True) for a in kg.find_all("a", class_="kwd-part")]
        if kws:
            return kws
    return meta_keywords


def parse_pubdate(meta):
    """citation_publication_date is YYYY/MM/DD. Return (year, month_name, raw)."""
    raw = meta.get("citation_publication_date") or meta.get("citation_online_date") or ""
    m = re.match(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", raw)
    if not m:
        return "", "", raw
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    return m.group(1), months[int(m.group(2)) - 1], raw


def parse_abstract(soup):
    """Structured abstract -> {section: text}; unstructured -> {'Abstract': text}."""
    sect = soup.find("section", class_="abstract")
    if sect is None:
        return {}
    out = {}
    secs = sect.find_all("div", class_="sec")
    if not secs:
        text = sect.get_text(" ", strip=True)
        return {"Abstract": text} if text else {}
    for sec in secs:
        title_el = sec.find("div", class_="title")
        para_el = sec.find("p", class_="chapter-para")
        title = title_el.get_text(strip=True) if title_el else "Abstract"
        text = para_el.get_text(" ", strip=True) if para_el else ""
        if text:
            out[title] = text
    return out


def parse_body(soup):
    """Walk the article body, returning (content, acknowledgments, key_messages, figures).

    IJE puts the body as a flat sequence of <h2>/<h3>/<p>/<div> inside <div class="widget-items">.
    We use h2 class to distinguish main body sections (`section-title`) from back-matter
    (`backsection-title`, `backreferences-title`, etc.) and acknowledgements.
    """
    intro = None
    for h2 in soup.find_all("h2", class_="section-title"):
        if h2.get_text(strip=True).lower() != "abstract":
            intro = h2
            break
    if intro is None:
        return {}, "", "", {}
    container = intro.parent

    content, figures = {}, {}
    key_messages, acknowledgments = "", ""
    cur_title, cur_kind, buf = None, None, []

    def commit():
        nonlocal acknowledgments, buf
        if cur_title and buf:
            text = "\n\n".join(buf).strip()
            if cur_kind == "body" and text:
                content[cur_title] = text
            elif cur_kind == "ack" and text:
                acknowledgments = text
        buf = []

    for child in container.children:
        if not getattr(child, "name", None):
            continue
        cls = set(child.get("class") or [])

        if child.name == "section" and "abstract" in cls:
            continue

        if child.name == "h2":
            commit()
            cur_title = child.get_text(strip=True)
            if "section-title" in cls and cur_title.lower() != "abstract":
                cur_kind = "body"
            elif "backacknowledgements-title" in cls:
                cur_kind = "ack"
            else:
                cur_kind = None  # references / author contributions / funding / etc.

        elif child.name == "h3" and "section-title" in cls and cur_kind == "body":
            buf.append(f"### {child.get_text(strip=True)}")

        elif child.name == "p" and "chapter-para" in cls and cur_kind in ("body", "ack"):
            buf.append(child.get_text(" ", strip=True))

        elif child.name == "div" and "boxed-text" in cls:
            txt = child.get_text(" ", strip=True)
            key_messages = re.sub(r"^Key Messages\s*", "", txt, flags=re.I).strip()

        elif child.name == "div" and "fig" in cls:
            label_el = child.find("div", class_="fig-label")
            cap_el   = child.find("div", class_="fig-caption")
            img_el   = child.find("img")
            label   = label_el.get_text(strip=True) if label_el else ""
            caption = cap_el.get_text(" ", strip=True) if cap_el else ""
            img_url = img_el.get("src") if img_el else ""
            if label and label not in figures:
                figures[label] = [caption, img_url]

        elif (child.name == "div"
              and "table-full-width-wrap" in cls
              and "table-modal" not in cls):
            label_el = child.find("span", class_="title-label")
            cap_el   = child.find("div", class_="caption")
            label   = label_el.get_text(strip=True) if label_el else ""
            caption = cap_el.get_text(" ", strip=True) if cap_el else ""
            if label and label not in figures:
                figures[label] = [caption, ""]

    commit()
    return content, acknowledgments, key_messages, figures


# ----------------------------- per-article ------------------------
def safe_segment(s):
    return re.sub(r"[^\w\-.]+", "_", s)


def process_article(article_url, out_dir):
    soup = BeautifulSoup(fetch(article_url), "html.parser")
    meta = parse_citation_meta(soup)

    doi = meta.get("citation_doi", "")
    if not doi:
        raise ValueError("missing DOI")
    title = meta.get("citation_title", "")
    pub_year, pub_month, pub_date = parse_pubdate(meta)

    doi_prefix = doi.split("/", 1)[0]
    doi_suffix = doi.split("/")[-1]
    folder = os.path.join(
        out_dir,
        safe_segment(doi_prefix),
        pub_year or "unknown",
        pub_month or "unknown",
        safe_segment(doi_suffix),
    )
    os.makedirs(folder, exist_ok=True)

    article_features = {
        "Title":     title,
        "URL":       article_url,
        "DOI":       doi,
        "PMID":      meta.get("citation_pmid", ""),
        "Published": pub_date,
        "Volume":    meta.get("citation_volume", ""),
        "Issue":     meta.get("citation_issue", ""),
        "Journal":   meta.get("citation_journal_title", ""),
        "Keywords":  parse_keywords(soup, meta["_meta_keywords"]),
        "Authors":   meta["_authors"],
        "PDF_URL":   meta.get("citation_pdf_url", ""),
    }
    with open(os.path.join(folder, "article_features.json"), "w", encoding="utf-8") as f:
        json.dump(article_features, f, indent=4, ensure_ascii=False)

    abstract = parse_abstract(soup)
    with open(os.path.join(folder, "abstract.json"), "w", encoding="utf-8") as f:
        json.dump(abstract, f, indent=4, ensure_ascii=False)

    content, acknowledgments, key_messages, figures = parse_body(soup)

    with open(os.path.join(folder, "content.json"), "w", encoding="utf-8") as f:
        json.dump(content, f, indent=4, ensure_ascii=False)

    if acknowledgments:
        with open(os.path.join(folder, "acknowledgments.txt"), "w", encoding="utf-8") as f:
            f.write(acknowledgments)
    if key_messages:
        with open(os.path.join(folder, "key_messages.txt"), "w", encoding="utf-8") as f:
            f.write(key_messages)
    if figures:
        with open(os.path.join(folder, "figure_table.json"), "w", encoding="utf-8") as f:
            json.dump(figures, f, indent=4, ensure_ascii=False)

    '''
    ### Optional: download figures and PDF
    src_dir = os.path.join(folder, "src")
    os.makedirs(src_dir, exist_ok=True)
    for label, (caption, url) in figures.items():
        if not url:
            continue
        try:
            r = creq.get(url, impersonate="chrome124", timeout=60)
            if r.status_code == 200:
                ext = os.path.splitext(url.split("?")[0])[1] or ".jpg"
                with open(os.path.join(src_dir, safe_segment(label) + ext), "wb") as f:
                    f.write(r.content)
        except Exception as e:
            print(f"  figure download failed ({label}): {e}")

    pdf_url = article_features["PDF_URL"]
    if pdf_url:
        try:
            r = creq.get(pdf_url, impersonate="chrome124", timeout=120)
            if r.status_code == 200:
                with open(os.path.join(src_dir, doi.split("/")[-1] + ".pdf"), "wb") as f:
                    f.write(r.content)
        except Exception as e:
            print(f"  pdf download failed: {e}")
    '''


# ----------------------------- main -------------------------------
def main():
    print(f"Target: IJE {TARGET_YEAR} {TARGET_MONTH or '(all issues)'}")
    issues = get_issue_links(TARGET_YEAR, TARGET_MONTH)
    if not issues:
        raise RuntimeError(
            f"No issues found for {TARGET_YEAR} matching month={TARGET_MONTH!r}"
        )
    print(f"Found {len(issues)} issue(s):")
    for label, url in issues:
        print(f"  {label} -> {url}")

    article_urls = []
    print("\nGathering article URLs...")
    for label, issue_url in tqdm(issues):
        urls = get_section_article_urls(issue_url, SECTION_TITLE)
        print(f"  {label}: {len(urls)} '{SECTION_TITLE}'")
        article_urls.extend(urls)
        time.sleep(random.uniform(1.0, 2.0))

    print(f"\nTotal {len(article_urls)} articles to crawl.\n")
    n_ok = 0
    for url in tqdm(article_urls):
        try:
            process_article(url, OUT_DIR)
            n_ok += 1
        except KeyboardInterrupt:
            print("Interrupted by user.")
            break
        except Exception as err:
            print(f"FAILED {url}: {err}")
        time.sleep(random.uniform(0.5, 1.5))

    print(f"\nSaved {n_ok}/{len(article_urls)} articles to {OUT_DIR}")


if __name__ == "__main__":
    main()
