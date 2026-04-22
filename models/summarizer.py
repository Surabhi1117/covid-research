import torch
from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

class Summarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.model_name = model_name
        self.summarizer = None

    def _load_model(self):
        if self.summarizer is None:
            logger.info(f"Loading summarization model {self.model_name}...")
            device = 0 if torch.cuda.is_available() else -1
            self.summarizer = pipeline("summarization", model=self.model_name, device=device)

    def summarize(self, text: str, max_length: int = 150, min_length: int = 40) -> str:
        """Generate a summary for the given text."""
        if not text or len(text.strip()) < 100:
            return text
        
        self._load_model()
        
        # Truncate text if too long for BART (usually 1024 tokens)
        truncated_text = text[:4000] 
        
        summary = self.summarizer(truncated_text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
