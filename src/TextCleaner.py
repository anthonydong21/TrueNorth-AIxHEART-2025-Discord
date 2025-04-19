# Simple PDF Document Cleaning
def clean_pdf_documents(docs, min_content_length=20, verbose=False) -> tuple:
    """
    Cleans a list of PDF document objects by removing:
    - Pages with very short content (e.g., subheaders, blank pages)
    - Duplicate pages based on exact text content

    Parameters:
        - docs (list): A list of document objects, each with `page_content` and `metadata`.
        - min_content_length (int): Minimum number of characters required for a page to be kept.
        - verbose (bool): If True, prints detailed logs of the cleaning process.

    Returns:
        tuple: (cleaned_docs, stats)
            - cleaned_docs (list): A list of cleaned document objects.
            - stats (dict): A dictionary with counts of total, removed, and remaining pages.
    """

    stats = {"total": len(docs), "short": 0, "duplicate": 0}

    # Dictionary to store unique content for duplicate detection
    content_map = {}

    # List to store cleaned documents
    cleaned_docs = []

    # Iterate through all document pages
    for i, doc in enumerate(docs):
        # Assign proper page numbering in metadata
        doc.metadata["page"] = i + 1

        if verbose:
            print("===Before cleaning: ")
            print(type(doc.page_content))
            print(doc.page_content)
            for char in doc.page_content:
                print(f"'{char}': {ord(char)}")

        # remove unicode misrepresentations
        # content = re.sub(r"[^a-zA-Z0-9\s.,!?;:'\"()\[\]{}-]", "", doc.page_content)
        content = doc.page_content.strip()
        content = str(content).replace(chr(65533), "ti")
        doc.page_content = content

        if verbose:
            print("===After cleaning: ")
            print(type(doc.page_content))
            print(doc.page_content)
            for char in doc.page_content:
                print(f"'{char}': {ord(char)}")

        # Check if content is too short (likely irrelevant)
        if len(content) < min_content_length:
            stats["short"] += 1  # count short pages
            if verbose:
                print(f"Skipping page {i+1}: Short content ({len(content)} chars)")
            continue  # Skip this page

        # Check for duplicates
        if content in content_map:
            stats["duplicate"] += 1  # Count duplicate pages
            if verbose:
                print(f"Skipping page {i+1}: Duplicate of page {content_map[content]+1}")
            continue  # Skip this page

        # Add unique content to tracking dictionary (avoid future duplicates)
        content_map[content] = i

        # Append the valid document to the cleaned list
        cleaned_docs.append(doc)

    # Calculate the number of valid pages after cleaning
    stats["valid"] = stats["total"] - stats["short"] - stats["duplicate"]

    # Print a cleaning summary if verbose mode is enabled
    if verbose:
        print(f"\nCleaning Summary:")
        print(f"  Total pages: {stats['total']}")
        print(f"  Removed: {stats['short']} short pages, {stats['duplicate']} duplicates")
        print(f"  Remaining: {stats['valid']} valid pages")

    # Return cleaned documents and summary statistics
    return cleaned_docs, stats
