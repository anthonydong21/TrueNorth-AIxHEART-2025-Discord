import re
from typing import List, Tuple, Dict


@staticmethod
def clean_documents(docs, min_content_length: int = 20, verbose: bool = False) -> Tuple[List, Dict]:
    """
    Cleans a list of PDF document objects by removing:
    - Pages with very short content
    - Duplicate pages based on exact text

    Preserves metadata from all documents, even removed ones.

    Returns:
        tuple:
            - cleaned_docs (list): Pages retained after filtering
            - stats (dict): Summary counts + detailed removed metadata
    """

    stats = {"total": len(docs), "short": 0, "duplicate": 0, "valid": 0, "removed_docs": []}  # ⬅️ Store metadata of removed pages

    cleaned_docs = []
    content_map = {}

    for i, doc in enumerate(docs):
        # Set explicit page number
        doc.metadata["page"] = i + 1

        if verbose:
            print(f"\n=== Before cleaning page {i+1} ===")
            print(repr(doc.page_content))

        # Normalize page content
        content = str(doc.page_content)

        # Replace Unicode replacement character (often �)
        content = content.replace(chr(65533), "ti")

        # Collapse excessive newlines and surrounding spaces
        content = re.sub(r"\s*\n\s*", " ", content)  # single-line mode
        content = re.sub(r"\n{2,}", "\n", content)  # multiple line breaks → one

        # Collapse extra spaces and strip leading/trailing whitespace
        content = re.sub(r"[ \t]{2,}", " ", content).strip()

        doc.page_content = content

        if verbose:
            print(f"=== After cleaning page {i+1} ===")
            print(repr(content))

        # Filter: too short
        if len(content) < min_content_length:
            stats["short"] += 1
            stats["removed_docs"].append({"reason": "short", "page": i + 1, "metadata": doc.metadata.copy()})
            continue

        # Filter: duplicate
        if content in content_map:
            stats["duplicate"] += 1
            stats["removed_docs"].append({"reason": "duplicate", "original_page": content_map[content] + 1, "page": i + 1, "metadata": doc.metadata.copy()})
            continue

        # Register unique content
        content_map[content] = i
        cleaned_docs.append(doc)

    stats["valid"] = len(cleaned_docs)

    if verbose:
        print(f"\n📋 Cleaning Summary:")
        print(f"  Total: {stats['total']}")
        print(f"  Removed: {stats['short']} short, {stats['duplicate']} duplicates")
        print(f"  Remaining: {stats['valid']}")
        print(f"  Removed metadata: {stats['removed_docs']}")

    return cleaned_docs, stats
