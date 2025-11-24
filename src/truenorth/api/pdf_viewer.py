"""
PDF page viewer API endpoint.
Serves PDF pages as images with page range support.
"""
import os
from pathlib import Path
from typing import Optional
from fastapi import HTTPException
import fitz  # PyMuPDF

# Base directory for PDFs
BOOKS_DIR = Path(__file__).parent.parent.parent / "books_pdf"


def get_pdf_path(filename: str) -> Path:
    """
    Securely get PDF path, preventing directory traversal.
    
    Args:
        filename: PDF filename (e.g., "Donaldson_van_Zyl_2022_PERMA4_Work_Related_Wellbeing.pdf")
    
    Returns:
        Path to PDF file
    
    Raises:
        HTTPException: If file not found or invalid
    """
    # Remove any path traversal attempts
    filename = os.path.basename(filename)
    
    # Construct full path
    pdf_path = BOOKS_DIR / filename
    
    # Verify file exists and is within BOOKS_DIR
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF not found")
    
    if not pdf_path.is_relative_to(BOOKS_DIR):
        raise HTTPException(status_code=403, detail="Invalid file path")
    
    return pdf_path


def render_pdf_pages(
    filename: str,
    page_num: int,
    range_before: int = 2,
    range_after: int = 2,
    dpi: int = 150,
    highlight_text: Optional[str] = None
) -> list[dict]:
    """
    Render PDF pages to images.
    
    Args:
        filename: PDF filename
        page_num: Target page number (1-indexed)
        range_before: Number of pages before target to include
        range_after: Number of pages after target to include
        dpi: Rendering DPI (higher = better quality but larger file)
        highlight_text: Text snippet to highlight on the target page
    
    Returns:
        List of dicts with page info:
        [
            {
                "page_num": 1,
                "image_base64": "...",
                "is_target": False
            },
            ...
        ]
    """
    pdf_path = get_pdf_path(filename)
    
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        # Calculate page range (1-indexed to 0-indexed)
        target_idx = page_num - 1
        start_idx = max(0, target_idx - range_before)
        end_idx = min(total_pages - 1, target_idx + range_after)
        
        pages_data = []
        
        for idx in range(start_idx, end_idx + 1):
            page = doc[idx]
            
            # Apply highlighting if this is the target page and text is provided
            if idx == target_idx and highlight_text:
                # Clean up search text (remove extra spaces)
                clean_text = " ".join(highlight_text.split())
                # Search for text occurrences
                text_instances = page.search_for(clean_text)
                
                # If exact match fails, try splitting into smaller chunks or words
                # For now, let's just stick to the full phrase or fall back to first few words if it's long
                if not text_instances and len(clean_text) > 50:
                     # Try searching for the first 10 words
                     short_text = " ".join(clean_text.split()[:10])
                     text_instances = page.search_for(short_text)

                if text_instances:
                    # Draw highlight shapes
                    shape = page.new_shape()
                    for rect in text_instances:
                        # Almond Silk color: #e6e6fa -> (0.90, 0.90, 0.98)
                        shape.draw_rect(rect)
                    
                    # Fill with semi-transparent background gradient purple
                    shape.finish(color=None, fill=(0.90, 0.90, 0.98), fill_opacity=0.5)
                    shape.commit()
            
            # Render page to pixmap
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PNG bytes
            img_bytes = pix.tobytes("png")
            
            # Encode to base64
            import base64
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            
            pages_data.append({
                "page_num": idx + 1,  # Convert back to 1-indexed
                "image_base64": img_base64,
                "is_target": (idx == target_idx),
                "width": pix.width,
                "height": pix.height
            })
        
        doc.close()
        return pages_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rendering PDF: {str(e)}")


def get_pdf_metadata(filename: str) -> dict:
    """
    Get PDF metadata.
    
    Args:
        filename: PDF filename
    
    Returns:
        Dict with metadata (title, author, page_count, etc.)
    """
    pdf_path = get_pdf_path(filename)
    
    try:
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        page_count = len(doc)
        doc.close()
        
        return {
            "filename": filename,
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "page_count": page_count,
            "format": metadata.get("format", ""),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading PDF metadata: {str(e)}")

