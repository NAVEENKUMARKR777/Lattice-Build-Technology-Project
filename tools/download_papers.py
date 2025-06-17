import argparse
import json
from datetime import datetime
from pathlib import Path

import arxiv
from tqdm import tqdm


def fetch_latest_quantum_papers(num_papers: int = 50):
    """Return a list of dicts containing title & abstract of latest `quant-ph` papers."""
    # arxiv API can return an empty page for very high indices; we'll fetch lazily and stop if that happens
    search = arxiv.Search(
        query="cat:quant-ph",
        max_results=num_papers,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers: list[dict] = []

    try:
        for result in tqdm(search.results(), desc="Downloading metadata"):
            papers.append(
                {
                    "id": result.get_short_id(),
                    "title": result.title.strip().replace("\n", " "),
                    "summary": result.summary.strip().replace("\n", " "),
                    "published": result.published.isoformat(),
                    "url": result.entry_id,
                }
            )

            if len(papers) >= num_papers:
                break  # safeguard in case API yields more entries than requested

    except arxiv.UnexpectedEmptyPageError:
        # This happens when the start index exceeds available results.
        print("[!] Reached end of available results early – returning", len(papers), "papers.")

    return papers


def main():
    parser = argparse.ArgumentParser(description="Download latest quantum physics papers from arXiv")
    parser.add_argument("-n", "--num_papers", type=int, default=50, help="Number of papers to fetch")
    parser.add_argument("-o", "--output_dir", type=str, default="data/papers", help="Directory to save JSON files")
    args = parser.parse_args()

    papers = fetch_latest_quantum_papers(args.num_papers)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = output_dir / f"quant_ph_{ts}.json"
    with outfile.open("w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(papers)} papers -> {outfile}")


if __name__ == "__main__":
    main() 