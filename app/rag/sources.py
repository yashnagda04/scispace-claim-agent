"""Find the same paper in PMC's official public PDF dataset."""

import re
from html import unescape
from urllib.parse import urlsplit
from xml.etree import ElementTree

PMC_IDS = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
PMC_CLOUD = "https://pmc-oa-opendata.s3.amazonaws.com/"


def normalize_doi(value):
    return re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", value.strip(), flags=re.I).lower()


def normalize_title(value):
    return "".join(c for c in re.sub(r"<[^>]+>", "", unescape(value)).casefold() if c.isalnum())


def find_pmc_ids(papers, client):
    """Use an explicit PMC link or batch-convert DOIs; never search by topic."""
    found = {}
    by_doi = {}
    for paper in papers:
        url = urlsplit(paper.get("PDF Link", ""))
        match = re.search(r"/articles/(PMC\d+)(?:/|$)", url.path)
        if url.hostname in {"www.ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov"} and match:
            found[paper["paper_id"]] = match[1]
        else:
            doi = normalize_doi(paper.get("DOI", ""))
            if doi:
                by_doi.setdefault(doi, []).append(paper["paper_id"])
    dois = list(by_doi)
    error = None
    for start in range(0, len(dois), 200):
        try:
            response = client.get(PMC_IDS, params={
                "ids": ",".join(dois[start:start + 200]), "idtype": "doi",
                "format": "json", "tool": "scispace_assignment",
            })
            response.raise_for_status()
            data = response.json()
            if data.get("status") != "ok":
                raise ValueError("PMC ID lookup was unsuccessful.")
            for record in data.get("records", []):
                doi = normalize_doi(record.get("doi", ""))
                pmcid = record.get("pmcid", "")
                if doi in by_doi and re.fullmatch(r"PMC\d+", pmcid) and record.get("live") != "false":
                    for paper_id in by_doi[doi]:
                        found[paper_id] = pmcid
        except Exception as exc:
            error = f"PMC ID lookup failed: {type(exc).__name__}."
    return found, error


def pmc_pdf_url(paper, pmcid, client):
    """Check identity and prefer a published PDF over an author manuscript."""
    response = client.get(PMC_CLOUD, params={"list-type": "2", "prefix": f"{pmcid}.", "delimiter": "/"})
    response.raise_for_status()
    listing = ElementTree.fromstring(response.content)
    if listing.findtext("{*}IsTruncated") == "true":
        raise ValueError("PMC version listing was incomplete.")
    prefixes = [node.text for node in listing.findall("{*}CommonPrefixes/{*}Prefix")]
    candidates = []
    for prefix in prefixes:
        if not re.fullmatch(rf"{pmcid}\.\d+/", prefix):
            continue
        response = client.get(f"{PMC_CLOUD}{prefix}{prefix.rstrip('/')}.json")
        response.raise_for_status()
        record = response.json()
        if record.get("pmcid") != pmcid:
            raise ValueError("PMC metadata has a different paper ID.")
        if normalize_title(record.get("title", "")) != normalize_title(paper["Paper Title"]):
            raise ValueError("PMC title does not match the supplied paper; PDF was not used.")
        doi = normalize_doi(paper.get("DOI", ""))
        if doi and normalize_doi(record.get("doi", "")) != doi:
            raise ValueError("PMC DOI does not match the supplied paper; PDF was not used.")
        url = urlsplit(record.get("pdf_url") or "")
        if url.scheme == "s3" and url.netloc == "pmc-oa-opendata" and url.path.startswith(f"/{prefix}"):
            manuscript = record.get("is_manuscript") in (True, "yes")
            candidates.append((manuscript, prefix, f"{PMC_CLOUD}{url.path.lstrip('/')}"))
    if not candidates:
        raise ValueError("No downloadable PDF in the PMC public dataset.")
    return sorted(candidates)[0][2]
