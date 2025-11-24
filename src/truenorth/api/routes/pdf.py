"""
FastAPI routes for PDF viewing.
"""
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
from truenorth.api.pdf_viewer import render_pdf_pages, get_pdf_metadata

router = APIRouter(prefix="/pdf", tags=["pdf"])


@router.get("/{filename}/pages")
async def get_pdf_pages(
    filename: str,
    page: int = Query(..., ge=1, description="Target page number (1-indexed)"),
    range_before: int = Query(2, ge=0, le=5, description="Pages before target"),
    range_after: int = Query(2, ge=0, le=5, description="Pages after target"),
    dpi: int = Query(150, ge=72, le=300, description="Rendering DPI"),
    highlight: str = Query(None, description="Text snippet to highlight on the target page")
):
    """
    Get rendered PDF pages as base64-encoded images.
    
    Returns a range of pages around the target page.
    """
    try:
        pages = render_pdf_pages(
            filename=filename,
            page_num=page,
            range_before=range_before,
            range_after=range_after,
            dpi=dpi,
            highlight_text=highlight
        )
        
        return JSONResponse(content={
            "pages": pages,
            "target_page": page,
            "filename": filename
        })
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{filename}/metadata")
async def get_metadata(filename: str):
    """
    Get PDF metadata (title, author, page count, etc.)
    """
    try:
        metadata = get_pdf_metadata(filename)
        return JSONResponse(content=metadata)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

