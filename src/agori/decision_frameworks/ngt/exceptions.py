class NGTError(Exception):
    """Base exception for NGT process errors"""

    pass


class ExpertGenerationError(NGTError):
    """Error during expert generation phase"""

    pass


class IdeaGenerationError(NGTError):
    """Error during idea generation phase"""

    pass


class SynthesisError(NGTError):
    """Error during results synthesis phase"""

    pass
