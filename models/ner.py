import spacy
import scispacy
import logging

logger = logging.getLogger(__name__)

class NERExtractor:
    def __init__(self, model_name: str = "en_ner_bc5cdr_md"):
        self.model_name = model_name
        self.nlp = None

    def _load_model(self):
        if self.nlp is None:
            logger.info(f"Loading NER model {self.model_name}...")
            try:
                self.nlp = spacy.load(self.model_name)
            except OSError:
                logger.error(f"Model {self.model_name} not found. Please run: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz")
                raise

    def extract_entities(self, text: str) -> dict:
        """Extract medical entities from text."""
        if not text:
            return {}
        
        self._load_model()
        doc = self.nlp(text)
        
        entities = {}
        for ent in doc.ents:
            label = ent.label_
            if label not in entities:
                entities[label] = set()
            entities[label].add(ent.text)
            
        # Convert sets to lists for JSON serialization
        return {k: list(v) for k, v in entities.items()}
