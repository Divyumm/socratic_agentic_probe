import re
from typing import Optional
from app_3.schemas import PapanekDimension, StudentState, Claim

class Teleprompter:
    """Translates raw system diagnostic messages and dimension names into supportive, student-friendly dialogue prompts."""

    BANNED_JARGON_MAP = {
        r"\bepistemic\b": "justification",
        r"\bvulnerability rank\b": "probing priority",
        r"\bcollapsed state\b": "need for a new perspective",
        r"\breasoning collapse\b": "new angle",
        r"\bplateau erosion\b": "exploring further",
        r"\b[rR]easoning [cC]ollapse\b": "new angle",
        r"\b[pP]lateau [eE]rosion\b": "exploring further"
    }

    DIMENSION_MASK_MAP = {
        PapanekDimension.METHOD: "technical implementation",
        PapanekDimension.USE: "usability",
        PapanekDimension.ASSOCIATION: "user perception",
        PapanekDimension.AESTHETICS: "visual design",
        PapanekDimension.TELESIS: "broader impact",
        PapanekDimension.NEED: "core user requirement",
    }

    def translate_prompt(self, raw_prompt: str, state: StudentState, 
                         dimension: PapanekDimension, emergent_theme: Optional[str] = None) -> str:
        """Transforms a raw assessor/router prompt into supportive, student-facing phrasings."""
        text = raw_prompt
        
        # 1. Strip system diagnostic brackets and fallback prefixes
        text = text.replace("[REASONING COLLAPSE DETECTED]\n", "")
        text = text.replace("[PLATEAU EROSION DETECTED - PROBING DEEPER]\n", "")
        text = text.replace("[GROUNDED RESOLUTION CONFIRMED]\n", "")
        text = re.sub(r"Fallback extracted claim \([^)]+\):\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"Fallback extracted claim:\s*", "", text, flags=re.IGNORECASE)
        
        # 2. Mask predefined Papanek dimensions or translate Emergent theme
        if dimension == PapanekDimension.EMERGENT:
            theme_phrase = emergent_theme if emergent_theme else "the situational context of your work"
            # Replace dimension phrasing
            text = re.sub(
                rf"\b{PapanekDimension.EMERGENT.value}\b(?:\s*dimension\b)?", 
                theme_phrase, 
                text, 
                flags=re.IGNORECASE
            )
        else:
            for dim, mask in self.DIMENSION_MASK_MAP.items():
                text = re.sub(
                    rf"\b{dim.value}\b(?:\s*dimension\b)?", 
                    mask, 
                    text, 
                    flags=re.IGNORECASE
                )

        # 3. Clean banned jargon using regex mapping
        for pattern, replacement in self.BANNED_JARGON_MAP.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            
        # 4. Standardise spacing and clean trailing newlines
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def translate_prompt_v2(self, raw_prompt: str, state: StudentState, 
                            claim: Optional[Claim] = None) -> str:
        """
        Phase 2 Design: Explicitly surfaces the underlying assumption from the claim 
        before asking the student to justify or revise it, rather than just doing a regex swap.
        """
        text = raw_prompt
        
        # 1. Strip system diagnostic brackets and fallback prefixes
        text = text.replace("[REASONING COLLAPSE DETECTED]\n", "")
        text = text.replace("[PLATEAU EROSION DETECTED - PROBING DEEPER]\n", "")
        text = text.replace("[GROUNDED RESOLUTION CONFIRMED]\n", "")
        text = re.sub(r"Fallback extracted claim \([^)]+\):\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"Fallback extracted claim:\s*", "", text, flags=re.IGNORECASE)
        
        # 2. Surface supportive state-based language (without restating the claim text)
        prefix = ""
        if claim:
            if state == StudentState.COLLAPSED:
                prefix += "Let's look at this from a new angle. "
            elif state == StudentState.UNSTABLE:
                prefix += "I'd like to explore the foundations of that a bit more. "
                
        # Combine the surfacing prefix with the cleansed raw prompt
        text = f"{prefix}{text}"

        # 3. Mask predefined Papanek dimension names (e.g. "Telesis", "Association") or
        # translate the Emergent theme. Probing questions are LLM/fixture-authored and
        # often name their own dimension directly, so this must run on the full text,
        # not just the wrapper prefix.
        if claim and claim.dimension == PapanekDimension.EMERGENT:
            theme_phrase = claim.emergent_theme if claim.emergent_theme else "the situational context of your work"
            text = re.sub(
                rf"\b{PapanekDimension.EMERGENT.value}\b(?:\s*dimension\b)?",
                theme_phrase,
                text,
                flags=re.IGNORECASE
            )
        else:
            for dim, mask in self.DIMENSION_MASK_MAP.items():
                text = re.sub(
                    rf"\b{dim.value}\b(?:\s*dimension\b)?",
                    mask,
                    text,
                    flags=re.IGNORECASE
                )

        # 4. Clean banned jargon using regex mapping (safety net)
        for pattern, replacement in self.BANNED_JARGON_MAP.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # 5. Standardise spacing
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
