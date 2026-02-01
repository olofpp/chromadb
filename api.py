from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import chromadb
import os

app = FastAPI(title="ChromaDB API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure ChromaDB data directory exists
CHROMA_DATA_PATH = "./chroma_data"
os.makedirs(CHROMA_DATA_PATH, exist_ok=True)

# Initialize ChromaDB client with persistent storage
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)


# Pydantic models
class Document(BaseModel):
    id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


class DocumentUpdate(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None


class DocumentBatch(BaseModel):
    ids: List[str]
    documents: List[str]
    metadatas: Optional[List[Dict[str, Any]]] = None


class QueryRequest(BaseModel):
    query_texts: List[str]
    n_results: int = 10
    where: Optional[Dict[str, Any]] = None
    where_document: Optional[Dict[str, Any]] = None


# Serve index.html
@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return HTMLResponse(content=content, headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        })
    return "<h1>index.html not found</h1>"


# Serve chat.html
@app.get("/chat", response_class=HTMLResponse)
@app.get("/chat.html", response_class=HTMLResponse)
async def chat():
    html_path = os.path.join(os.path.dirname(__file__), "chat.html")
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return HTMLResponse(content=content, headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        })
    return "<h1>chat.html not found</h1>"


# API Health check
@app.get("/health")
async def health():
    return {"message": "ChromaDB FastAPI Server", "status": "running"}


# Collection endpoints
@app.get("/collections")
async def list_collections():
    """List all collections"""
    try:
        collections = client.list_collections()
        return {"collections": [col.name for col in collections]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/collections/{collection_name}")
async def create_collection(collection_name: str):
    """Create a new collection or get existing one"""
    try:
        collection = client.get_or_create_collection(name=collection_name)
        return {"message": f"Collection '{collection_name}' ready", "name": collection.name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/collections/{collection_name}")
async def get_collection(collection_name: str):
    """Get collection details"""
    try:
        collection = client.get_collection(name=collection_name)
        count = collection.count()
        return {
            "name": collection.name,
            "count": count
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Collection not found: {str(e)}")


@app.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """Delete a collection"""
    try:
        client.delete_collection(name=collection_name)
        return {"message": f"Collection '{collection_name}' deleted"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


# Document endpoints
@app.post("/collections/{collection_name}/documents")
async def add_document(collection_name: str, doc: Document):
    """Add a single document to a collection"""
    try:
        collection = client.get_collection(name=collection_name)
        collection.add(
            ids=[doc.id],
            documents=[doc.content],
            metadatas=[doc.metadata] if doc.metadata else None
        )
        return {"message": "Document added", "id": doc.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/collections/{collection_name}/documents/{document_id}")
async def update_document(collection_name: str, document_id: str, doc: DocumentUpdate):
    """Update an existing document in a collection"""
    try:
        collection = client.get_collection(name=collection_name)
        collection.update(
            ids=[document_id],
            documents=[doc.content],
            metadatas=[doc.metadata] if doc.metadata else None
        )
        return {"message": "Document updated", "id": document_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/collections/{collection_name}/documents/batch")
async def add_documents_batch(collection_name: str, batch: DocumentBatch):
    """Add multiple documents to a collection"""
    try:
        collection = client.get_collection(name=collection_name)
        collection.add(
            ids=batch.ids,
            documents=batch.documents,
            metadatas=batch.metadatas
        )
        return {"message": f"{len(batch.ids)} documents added"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/collections/{collection_name}/documents")
async def get_documents(
    collection_name: str,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    where: Optional[str] = None
):
    """Get all documents from a collection with optional filtering"""
    try:
        collection = client.get_collection(name=collection_name)
        
        # Parse where filter if provided
        where_filter = None
        if where:
            import json
            where_filter = json.loads(where)
        
        result = collection.get(
            limit=limit,
            offset=offset,
            where=where_filter
        )
        
        # Format the response
        documents = []
        for i in range(len(result['ids'])):
            doc = {
                "id": result['ids'][i],
                "document": result['documents'][i] if result['documents'] else None,
                "metadata": result['metadatas'][i] if result['metadatas'] else None
            }
            documents.append(doc)
        
        return {
            "count": len(documents),
            "documents": documents
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/collections/{collection_name}/documents/{document_id}")
async def get_document(collection_name: str, document_id: str):
    """Get a specific document by ID"""
    try:
        collection = client.get_collection(name=collection_name)
        result = collection.get(ids=[document_id])
        if not result['ids']:
            raise HTTPException(status_code=404, detail="Document not found")
        return {
            "id": result['ids'][0],
            "document": result['documents'][0] if result['documents'] else None,
            "metadata": result['metadatas'][0] if result['metadatas'] else None
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/collections/{collection_name}/documents/{document_id}")
async def delete_document(collection_name: str, document_id: str):
    """Delete a document from a collection"""
    try:
        collection = client.get_collection(name=collection_name)
        collection.delete(ids=[document_id])
        return {"message": f"Document '{document_id}' deleted"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Query endpoints
@app.post("/collections/{collection_name}/query")
async def query_collection(collection_name: str, query: QueryRequest):
    """Query a collection with semantic search"""
    try:
        collection = client.get_collection(name=collection_name)
        results = collection.query(
            query_texts=query.query_texts,
            n_results=query.n_results,
            where=query.where,
            where_document=query.where_document
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
