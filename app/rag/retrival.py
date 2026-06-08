from langchain.chat_models import init_chat_model
from app.core.config import settings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.rag.pipeline import query_rag
from app.rag.qdrant import QdrantService
import base64
from app.core.logger import get_logger
from langchain_google_genai import ChatGoogleGenerativeAI


logger = get_logger(__name__)


SYSTEM_PROMPT = """You are a financial analyst assistant.
 
Answer ONLY from the provided document context (text excerpts and charts below).
 
Rules:
- Use both text excerpts and chart images to form your answer.
- Cite section names and page numbers when relevant.
- Be precise with figures and percentages — do not round or paraphrase numbers.
- If the information is not in the provided context, reply exactly:
  "I could not find this information in the document."
"""



class RetrivalService:
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=settings.GOOGLE_API_KEY,
        )
        self.qdrant_svc = QdrantService()
    
    async def query_rag_with_answer(self,query:str):

        # Retrive 
        results = await self.qdrant_svc.similarity_search_multimodal(
            query=query,
            text_k=5,
            image_k=3,
            text_threshold=0.5,
            image_threshold=0.3 
        )
        text_results = results["text_results"]
        image_results = results["image_results"]

        # Build prompt for LLM
        system_message = SystemMessage(content=SYSTEM_PROMPT)

        content = []
        content.append({
            "type": "text",
            "text": f"Question: {query}"
        })

        # Text Context
        if text_results:
            excerpts = []
            for i, r in enumerate(text_results, start=1):
                section = r["metadata"].get("section_title", "Unknown")
                page    = r["metadata"].get("page_num", "?")
                excerpts.append(
                    f"[Excerpt {i} | Section: {section} | Page: {page} | Score: {r['score']}]\n"
                    f"{r['text']}"
                )
 
            content.append({
                "type": "text",
                "text": "## Relevant text excerpts\n\n" + "\n\n---\n\n".join(excerpts),
            })


        # Images
        if image_results:
            content.append({
                "type": "text",
                "text": "## Relevant charts from the document",
            })
 
            for i, r in enumerate(image_results, start=1):
                b64  = r["metadata"].get("base64_image", "")
                mime = r["metadata"].get("mime_type", "image/png")
                page = r["metadata"].get("page_num", "?")
 
                if not b64:
                    logger.warning(f"Image result {i} missing base64_image, skipping.")
                    continue
 
                # Label before the image so Gemini knows what it's looking at
                content.append({
                    "type": "text",
                    "text": f"[Chart {i} | Page: {page} | Score: {r['score']}]",
                })
 
                # Image block — data-URI format required by langchain-google-genai
                # "data:{mime};base64,{b64}" is the standard data-URI scheme
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{b64}",
                    },
                })
        human_message = HumanMessage(content=content)
        
        response = await self.model.ainvoke([system_message, human_message])
        return response
    

      