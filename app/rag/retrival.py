from langchain.chat_models import init_chat_model
from app.core.config import settings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.rag.pipeline import query_rag
from app.rag.qdrant import QdrantService
import base64
from app.core.logger import get_logger
from langchain_google_genai import ChatGoogleGenerativeAI


logger = get_logger(__name__)


SYSTEM_PROMPT = """You are a financial analyst assistant specializing in Indian mutual fund factsheets.

Answer ONLY from the provided document context. Context chunks are labelled with:
  [Excerpt N | Fund: <name> | Type: <type> | Page: <page> | Score: <score>]
where Type is one of:
  - returns          : CAGR performance table (1Y / 3Y / 5Y / Since Inception)
  - portfolio        : Full list of equity / debt holdings with % NAV
  - scheme_details   : AUM, NAV, expense ratio, fund manager, exit load
  - quant_indicators : Std Dev, Sharpe Ratio, Portfolio Beta, Dividend Yield
  - summary_box      : Top 5 stock/sector rankings
  - text             : Fund description, footnotes, style box

Rules:
- Use exact numbers — do not round or paraphrase figures.
- When a returns table is present, use its 1Y / 3Y / 5Y / SI columns directly.
- Cite Fund name and page number when you quote a figure.
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
    
    async def query_rag_with_answer(self, query: str):

        # Retrieve
        results = await self.qdrant_svc.similarity_search_multimodal(
            query=query,
            text_k=10,           # enough to cover all table types for 2+ funds
            image_k=3,
            text_threshold=0.45,
            image_threshold=0.3,
        )
        text_results = results["text_results"]
        image_results = results["image_results"]

        # Build prompt for LLM
        system_message = SystemMessage(content=SYSTEM_PROMPT)
        content: list = [{"type": "text", "text": f"Question: {query}"}]

        # ── Text / structured-table context ──
        if text_results:
            excerpts = []
            seen_texts: set[str] = set()   # dedup identical chunks
            for i, r in enumerate(text_results, start=1):
                text = r["text"]
                if text in seen_texts:
                    continue
                seen_texts.add(text)

                meta = r["metadata"]
                fund      = meta.get("fund_name") or meta.get("section_title") or "Unknown"
                ttype     = meta.get("table_type") or meta.get("modality") or "text"
                page      = meta.get("page_num", "?")
                month     = meta.get("month_year", "")
                score     = r["score"]

                label = (
                    f"[Excerpt {i} | Fund: {fund} | Type: {ttype}"
                    + (f" | Month: {month}" if month else "")
                    + f" | Page: {page} | Score: {score}]"
                )
                excerpts.append(f"{label}\n{text}")

            if excerpts:
                content.append({
                    "type": "text",
                    "text": "## Document context\n\n" + "\n\n---\n\n".join(excerpts),
                })

        # ── Image context (only when images exist in collection) ──
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

                content.append({
                    "type": "text",
                    "text": f"[Chart {i} | Page: {page} | Score: {r['score']}]",
                })
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                })

        human_message = HumanMessage(content=content)
        response = await self.model.ainvoke([system_message, human_message])
        return response