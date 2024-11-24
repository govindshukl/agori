import asyncio
import logging

# import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, TypedDict

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI

from .exceptions import (  # NGTError,
    ExpertGenerationError,
    IdeaGenerationError,
    SynthesisError,
)
from .models import Config, NGTExpert, NGTResponse
from .prompts import aggregation_prompt, analysis_prompt, expert_generation_prompt

log = logging.getLogger(__name__)


class SectionContent(TypedDict):
    """Type definition for section content"""

    Key_Findings: List[str]
    Technical_Analysis: List[str]
    Actionable_Recommendations: List[str]
    Risk_Assessment: List[str]


class NGTProcessor:
    def __init__(
        self,
        deployment: str,
        endpoint: str,
        api_key: str,
        api_version: str,
        config: Config = None,
        verbose: bool = False,
    ):
        self.config = config or Config()
        self.chunks: List[str] = []
        self.executor = ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.config.VERBOSE = verbose
        self.logger = self._setup_logger()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=self._measure_tokens,  # Use token-based length
            separators=[
                "\n\n",  # Paragraphs
                "\n",  # Lines
                ". ",  # Sentences
                ", ",  # Clauses
                " ",  # Words
                "",  # Characters
            ],
        )

        self.llm = AzureChatOpenAI(
            azure_deployment=deployment,
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            temperature=1,
        )

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("NGTProcessor")
        if self.config.VERBOSE:
            logger.setLevel(logging.INFO)

            # Remove any existing handlers
            logger.handlers = []

            # Add console handler
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(message)s")  # Simplified format
            handler.setFormatter(formatter)
            logger.addHandler(handler)

            # Prevent log propagation to avoid duplicate messages
            logger.propagate = False

        return logger

    def _get_context(self) -> str:
        """Get concatenated context from all chunks."""
        return "\n\n".join(self.chunks)

    def add_documents(self, documents: List[Any]) -> None:
        """Add and chunk documents based on config settings."""
        text_content = "\n\n".join(doc.page_content for doc in documents)
        self.chunks = self.text_splitter.split_text(text_content)

    def _measure_tokens(self, text: str) -> int:
        """Measure token count of text."""
        return len(self.tokenizer.encode(text))

    async def generate_experts(self, query: str) -> List[NGTExpert]:
        """Generate expert roles only."""
        try:
            context = self._get_context()
            chain = expert_generation_prompt | self.llm
            context_truncated = context[:4000]

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                lambda: chain.invoke({"context": context_truncated, "query": query}),
            )

            experts = self._parse_expert_response(response)
            return experts

        except Exception as e:
            raise ExpertGenerationError(f"Expert generation failed: {str(e)}")

    async def analyze_with_experts(
        self, experts: List[NGTExpert], query: str
    ) -> List[NGTResponse]:
        """Generate responses from provided experts."""
        if self.config.VERBOSE:
            self.logger.info("\nStarting expert analysis phase...")
            total_experts = len(experts)

        expert_responses = []
        for idx, expert in enumerate(experts, 1):
            if self.config.VERBOSE:
                self.logger.info(
                    f"\nProcessing Expert {idx}/{total_experts}: {expert.role}"
                )

            response = await self.generate_ideas(expert, query)
            expert_responses.append(response)

            if self.config.VERBOSE:
                self.logger.info("\n=== Expert Analysis Results ===")
                self.logger.info(f"Expert: {expert.role}")
                self.logger.info("Analysis Summary:")
                self.logger.info(f"{response.ideas}")
                self.logger.info("\nMetrics:")
                for metric, value in response.metrics.items():
                    self.logger.info(f"{metric}: {value}")
                self.logger.info("=== End of Analysis ===\n")

        return expert_responses

    async def _process_single_chunk(
        self, chunk: str, expert: NGTExpert, query: str, chain: Any
    ) -> str:
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: chain.invoke(
                    {
                        "role": expert.role,
                        "specialty": expert.specialty,
                        "description": expert.description,
                        "text": chunk,
                        "query": query,
                    }
                ),
            )
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            log.error(f"Error processing chunk: {str(e)}")
            return f"Error processing chunk: {str(e)}"

    async def run_ngt_analysis(self, query: str) -> str:
        """Complete NGT analysis process."""
        if self.config.VERBOSE:
            self.logger.info("\n=== Starting NGT Analysis Process ===")
            self.logger.info(f"Query: {query}")

        # Step 1: Generate experts
        if self.config.VERBOSE:
            self.logger.info("\nPhase 1: Expert Generation")
        experts = await self.generate_experts(query)

        # Step 2: Generate expert responses
        if self.config.VERBOSE:
            self.logger.info("\nPhase 2: Expert Analysis")
        expert_responses = await self.analyze_with_experts(experts, query)

        # Step 3: Synthesize results
        if self.config.VERBOSE:
            self.logger.info("\nPhase 3: Synthesis")
        final_synthesis = await self.synthesize_ngt_results(expert_responses)

        if self.config.VERBOSE:
            self.logger.info("\n=== NGT Analysis Complete ===")

        return final_synthesis

    async def generate_ideas(self, expert: NGTExpert, query: str) -> NGTResponse:
        """Generate ideas processing chunks in parallel based on config."""
        try:
            chain = analysis_prompt | self.llm
            chunk_batches = [
                self.chunks[i : i + self.config.MAX_PARALLEL_REQUESTS]
                for i in range(0, len(self.chunks), self.config.MAX_PARALLEL_REQUESTS)
            ]

            ideas = []
            for batch in chunk_batches:
                batch_ideas = await asyncio.gather(
                    *[
                        self._process_single_chunk(chunk, expert, query, chain)
                        for chunk in batch
                    ]
                )
                ideas.extend(batch_ideas)

                total_tokens = sum(self._measure_tokens(idea) for idea in batch_ideas)
                cost = (total_tokens / 1000) * self.config.GPT4_COST_PER_1K_OUTPUT
                log.info(f"Batch processing cost estimate: ${cost:.4f}")

            return self._create_expert_response(expert, ideas)

        except Exception as e:
            raise IdeaGenerationError(f"Idea generation failed: {str(e)}")

    async def synthesize_ngt_results(self, expert_responses: List[NGTResponse]) -> str:
        """Synthesize NGT expert responses into final recommendations.

        Args:
            expert_responses (List[NGTResponse]): List of expert responses to synthesize

        Returns:
            str: Synthesized recommendations and analysis

        Raises:
            SynthesisError: If synthesis process fails
        """
        try:
            if self.config.VERBOSE:
                self.logger.info("\nStarting synthesis phase...")

            context = self._get_context()
            structured_responses = self._prepare_responses_for_synthesis(
                expert_responses
            )

            chain = aggregation_prompt | self.llm
            loop = asyncio.get_event_loop()

            # Get response from chain
            raw_response = await loop.run_in_executor(
                self.executor,
                lambda: chain.invoke(
                    {"responses": structured_responses, "context": context[:4000]}
                ),
            )

            # Ensure we get a string response
            if hasattr(raw_response, "content"):
                response_text = str(raw_response.content)
            else:
                response_text = str(raw_response)

            if not isinstance(response_text, str):
                raise SynthesisError(
                    "Failed to get string response from synthesis chain"
                )

            if self.config.VERBOSE:
                self.logger.info("\n=== Final Synthesis ===")
                self.logger.info(f"Length: {len(response_text)} characters")
                self.logger.info("=== End of Synthesis ===\n")

            return response_text

        except Exception as e:
            log.exception(f"Error in NGT synthesis: {str(e)}")
            raise SynthesisError(f"Results synthesis failed: {str(e)}")

        except Exception as e:
            log.exception(f"Error in NGT synthesis: {str(e)}")
            raise SynthesisError(f"Results synthesis failed: {str(e)}")

        except Exception as e:
            log.exception(f"Error in NGT synthesis: {str(e)}")
            raise SynthesisError(f"Results synthesis failed: {str(e)}")

    def _prepare_responses_for_synthesis(self, responses: List[NGTResponse]) -> str:
        """Prepare responses for synthesis phase.

        Args:
            responses (List[NGTResponse]): List of expert responses to be synthesized

        Returns:
            str: A formatted string containing all expert responses
        """
        # Initialize an empty list to store formatted responses
        formatted_responses: List[str] = []

        # Format each response
        for response in responses:
            formatted_response: str = (
                f"\nExpert: {response.expert.role}\n"
                f"Specialty: {response.expert.specialty}\n"
                f"Ideas and Analysis:\n"
                f"{response.ideas}\n"
                f"---"
            )
            formatted_responses.append(formatted_response)

        # Join all formatted responses with newlines and return
        return "\n".join(formatted_responses)

    def _parse_expert_response(self, response) -> List[NGTExpert]:
        """Parse LLM response into NGTExpert objects."""
        experts: List[NGTExpert] = []
        current_expert: Dict[str, str] = {}

        response_text = (
            response.content if hasattr(response, "content") else str(response)
        )

        for line in response_text.split("\n"):
            if line.strip():
                if line.startswith("Role:"):
                    if current_expert:
                        experts.append(NGTExpert(**current_expert))
                        current_expert = {}
                    current_expert["role"] = line.split("Role:")[1].strip()
                elif line.startswith("Description:"):
                    current_expert["description"] = line.split("Description:")[
                        1
                    ].strip()
                elif line.startswith("Specialty:"):
                    current_expert["specialty"] = line.split("Specialty:")[1].strip()

        if current_expert:
            experts.append(NGTExpert(**current_expert))

        return experts

    async def _process_chunks_for_ideas(
        self,
        expert: NGTExpert,
        chunks: List[Any],
        query: str,
        chain: Any,
        batch_size: int,
    ) -> List[str]:
        """Process document chunks to generate ideas."""

        async def process_single_chunk(chunk: Any) -> str:
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: chain.invoke(
                        {
                            "role": expert.role,
                            "specialty": expert.specialty,
                            "description": expert.description,
                            # Changed from chunk.content to chunk.page_content
                            "text": chunk.page_content,
                            "query": query,
                        }
                    ),
                )
                return (
                    response.content if hasattr(response, "content") else str(response)
                )
            except Exception as e:
                log.error(f"Error processing chunk: {str(e)}")
                return f"Error processing chunk: {str(e)}"

        # Process chunks in batches
        ideas: List[str] = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            batch_ideas = await asyncio.gather(
                *[process_single_chunk(chunk) for chunk in batch]
            )
            ideas.extend(batch_ideas)

            # Log progress
            progress = (i + len(batch)) / len(chunks) * 100
            log.info(f"Ideas Generation Progress: {progress:.1f}% complete")

        return ideas

    def _create_expert_response(
        self, expert: NGTExpert, ideas: List[str]
    ) -> NGTResponse:
        """Create structured NGTResponse from ideas."""
        sections: Dict[str, List[str]] = {
            "Key Findings": [],
            "Technical Analysis": [],
            "Actionable Recommendations": [],
            "Risk Assessment": [],
        }

        # Extract and organize content by section
        for idea in ideas:
            current_section: str | None = None
            for line in idea.split("\n"):
                for section in sections:
                    if section in line:
                        current_section = section
                        break
                if current_section and line.strip() and section not in line:
                    sections[current_section].append(line.strip())

        # Combine into structured format
        structured_content: List[str] = ["# Expert Analysis"]
        for section, content in sections.items():
            if content:
                structured_content.append(f"\n## {section}")
                structured_content.extend(list(set(content)))  # Remove duplicates

        # Create metrics
        metrics: Dict[str, Any] = {
            "total_ideas": len(ideas),
            "sections_covered": sum(1 for section in sections.values() if section),
            "ideas_per_section": {
                section: len(content) for section, content in sections.items()
            },
        }

        return NGTResponse(
            expert=expert, ideas="\n".join(structured_content), metrics=metrics
        )
