"""
Agori Decision Frameworks
------------------------
A collection of AI-powered decision-making frameworks.

Currently supported frameworks:
- NGT (Nominal Group Technique): A structured decision-making process that combines
  individual ideation with group discussion and voting.
- Delphi: An AI-powered implementation of the Delphi method for structured consensus
  building through iterative expert consultation.
"""

# from agori.decision_frameworks.ngt.exceptions import (
#     ExpertGenerationError,
#     IdeaGenerationError,
#     NGTError,
#     SynthesisError,
# )
# from agori.decision_frameworks.ngt.models import NGTExpert, NGTIdea, NGTResponse
from agori.decision_frameworks.ngt.processor import NGTProcessor

# from agori.decision_frameworks.delphi.exceptions import (
#     DelphiError,
#     RoundExecutionError,
#     ConvergenceError,
#     ExpertPanelError,
# )
# from agori.decision_frameworks.delphi.models import (
#     DelphiConfig,
#     DelphiExpert,
#     DelphiResponse,
#     DelphiReport,
#     Opinion,
# )
# from agori.decision_frameworks.delphi.processor import DelphiProcessor

# Version of the decision_frameworks package
__version__ = "0.1.0"

# Framework registry
AVAILABLE_FRAMEWORKS = {
    "ngt": {
        "name": "Nominal Group Technique",
        "version": "0.1.0",
        "description": (
            "AI-powered implementation of the Nominal Group Technique "
            "for structured decision making"
        ),
        "processor": NGTProcessor,
    },
    # "delphi": {
    #     "name": "Delphi Method",
    #     "version": "0.1.0",
    #     "description": (
    #         "AI-powered implementation of the Delphi method "
    #         "for structured consensus building"
    #     ),
    #     "processor": DelphiProcessor,
    # }
}
