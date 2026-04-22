import os
import google.generativeai as genai
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class RAGPipeline:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            # Use the correct model identifier
            self.model = genai.GenerativeModel('gemini-flash-latest')
        else:
            self.model = None
            print("Warning: GEMINI_API_KEY not found. RAG functionality will be disabled.")

    def answer_question(self, question: str, context_papers: List[Dict[str, Any]]) -> str:
        """
        Answer a question based on a list of retrieved papers.
        """
        if not self.model:
            return "Error: LLM API key not configured."

        if not context_papers:
            return "I couldn't find any relevant papers to answer your question."

        # Construct context string
        context_str = ""
        for i, paper in enumerate(context_papers, 1):
            title = paper.get("title", "Unknown Title")
            abstract = paper.get("abstract", "No abstract available")
            doi = paper.get("doi", "No DOI")
            context_str += f"[{i}] Title: {title}\nDOI: {doi}\nAbstract: {abstract}\n\n"

        prompt = f"""
You are a highly capable COVID-19 research assistant. Below is a set of research paper excerpts.
Use ONLY the provided context to answer the user's question. 
Always cite the papers using their numbers (e.g., [1], [2]) when making a claim.
If the context doesn't contain enough information to answer, say so.

CONTEXT:
{context_str}

USER QUESTION: {question}

SCIENTIFIC ANSWER:
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                return "Error: The AI model 'gemini-flash-latest' was not found. Please check your API key and model access."
            return f"Error generating answer: {str(e)}"

    def draft_paper(self, topic: str, context_papers: List[Dict[str, Any]]) -> str:
        """
        Draft a full research paper based on the retrieved context.
        """
        if not self.model:
            return "Error: LLM API key not configured."

        if not context_papers:
            return "I couldn't find enough relevant papers to draft a research paper on this topic."

        context_str = ""
        for i, paper in enumerate(context_papers, 1):
            title = paper.get('title', 'Unknown Title')
            abstract = paper.get('abstract', '')
            context_str += f"[{i}] Title: {title}\nAbstract: {abstract}\n\n"

        prompt = f"""
You are a world-class COVID-19 researcher. Based ONLY on the provided context, write a comprehensive, professional research paper draft on the topic: "{topic}".

The paper must include the following sections:
1. Title (Creative and academic)
2. Abstract (Concise summary)
3. Introduction (Background and significance)
4. Literature Review (Synthesizing the provided context with citations like [1], [2])
5. Discussion (Implications of the findings)
6. Conclusion (Summary and future outlook)
7. References (List the source papers)

CONTEXT:
{context_str}

SCIENTIFIC DRAFT:
"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                return "Error: The AI model 'gemini-1.5-flash' was not found. Please check your API key and model access."
            return f"Error generating draft: {str(e)}"
