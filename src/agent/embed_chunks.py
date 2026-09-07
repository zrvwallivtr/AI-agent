import hashlib

from src.config.models import EMBED_MODEL, MODEL_MAX_TOKENS
from src.agent.models.embed import Embed
from src.agent.tokenizers import Tknizr
from src.agent.models.validation import validate_model
from models_database import EMB_MODEL_DIMENSION


class EmbedChunks:
    def __init__(self, model: str | None = None):
        model = EMBED_MODEL if model is None else model
        #validate_model(model)

        if MODEL_MAX_TOKENS:
            self.tknizr = Tknizr(model, MODEL_MAX_TOKENS)
        else:
            self.tknizr = Tknizr(model)

        self.model      = model
        self.emb_dim    = EMB_MODEL_DIMENSION[self.model]

        # ===========================================
        # \\ Need to change to using the tokenizer \\
        # ===========================================
        self.chnk_size  = min(int(self.tknizr.model_max_tokens * 0.5), 500) # Set limit as half of the max tokens, cap at 500 tokens
        self.chnk_ovrlp = int(self.chnk_size * 0.15) # Overlap between consecutive chunks

        self.embed = Embed()


    # ====================================================================
    # PROCESSING TEXT
    # ====================================================================

    def _split_on_separator(self, txt: str, sep: str) -> list[str]:
        """Split text on a separator, keeping the separator's semantic boundary intact."""
        return [s for s in txt.split(sep) if s.strip()]


    # ====================================================================
    # CHUNKING FUNCTIONS
    # ====================================================================

    def _recursive_chunking_by_separator(self, txt: str, seps: list[str]) -> list[str]:
        """
        Recursively split text using priority list of separators. Every chunk
        must be under the set limit of 'self.chnk_size'.
        """
        sep, *rest_seps = seps # Get the first separator on the list
        pieces = self._split_on_separator(txt, sep)

        # === CHUNKING BY CURRENT SEPARATOR ==================================

        chnks = []
        buffer = ""

        for piece in pieces:
            # Add next piece onto the accumulated, and re-insert the separator in between
            cand = (buffer + sep + piece) if buffer else piece

            encoded = self.tknizr.encode_text(cand)
            if encoded:
                if len(encoded) <= self.chnk_size:
                    buffer = cand

                else:
                    # Flush already accumulated as finished chunk
                    if buffer:
                        chnks.append(buffer)
                    
                    # Run function again with the next separator until 'piece' is under the limit
                    encoded_piece = self.tknizr.encode_text(piece)
                    if encoded_piece:
                        if len(encoded_piece) > self.chnk_size:
                            chnks.extend(self._recursive_chunking_by_separator(piece, rest_seps)) # Try the next separator in 'seps'
                            buffer = ""

                        # 'piece' is under the limit
                        else:
                            buffer = piece

                    # Fallback if encoding failed
                    else:
                        buffer = piece

        # Flush rest of the buffer as finished chunk
        if buffer:
            chnks.append(buffer)
        
        return chnks


    def _add_overlap_to_chunks(self, chnks: list[str]) -> list[str]:
        """Overlap the content at the end from the previous chunk to prevent context lost."""
        # If there is nothing to overlap (One or no chunk)
        if len(chnks) <= 1:
            return chnks

        ovrlped = [chnks[0]] # No overlap for the first chunk

        # Every chunk after the first
        for i in range(1, len(chnks)):
            prev_words = chnks[i - 1].split() # Split previous chunk
            prev_tail = (
                " ".join(prev_words[-self.chnk_ovrlp:]) # Add preset previous overlap amount
                if len(prev_words) > self.chnk_ovrlp
                else chnks[i - 1]
            )
            ovrlped.append(f"{prev_tail} {chnks[i]}")
        return ovrlped


    def paragraph_chunking(self, cont: str) -> list[str]:
        """
        Split content into paragraph chunks with preset overlap, optimal for structured documents.

        Separator priority:
        1. Paragraph breaks
        2. Sentence breaks
        3. Word breaks
        4. Hard cut
        """
        seps = [
            "\n\n", # Paragraph
            ". ", # Sentence
            " " # Word
        ]
        chnks = self._recursive_chunking_by_separator(cont.strip(), seps)
        return self._add_overlap_to_chunks(chnks)
