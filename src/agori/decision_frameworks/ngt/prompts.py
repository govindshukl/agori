# ruff: noqa: E501
# pylint: disable=line-too-long,too-many-lines
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# Expert Generation Prompt
expert_generation_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "You are an NGT facilitator selecting expert roles."
        ),
        HumanMessagePromptTemplate.from_template(
            """
        Based on the following document excerpt and user query, identify 3 most
        relevant expert roles for NGT decision making.
        Document excerpt: {context}
        User query: {query}
        Return exactly 3 experts in this format:
        Role: [title]
        Description: [one sentence description]
        Specialty: [specific focus area]
        Consider:
        1. Domain expertise needed
        2. Different perspectives required
        3. Technical and non-technical aspects
        4. Implementation considerations
    """
        ),
    ]
)


# Analysis and Insights Prompt
analysis_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "You are a {role} with expertise in {specialty}. {description}"
        ),
        HumanMessagePromptTemplate.from_template(
            """
                    Based on the following content and query, provide expert analysis.
                    Content: {text}
                    Query: {query}
                    Provide a structured analysis with:
                    1. Key Findings: Most important discoveries
                    2. Technical Analysis: Detailed examination
                    3. Actionable Recommendations: Specific next steps
                    4. Risk Assessment: Potential challenges
                    Focus on your expertise area and be specific.
                """
        ),
    ]
)

# Idea Generation Prompt
idea_generation_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "You are a {role} with expertise in {specialty}. {description}"
        ),
        HumanMessagePromptTemplate.from_template(
            """
        Based on the following content and query, generate specific ideas and solutions.
        Content: {text}
        Query: {query}
        Follow NGT principles:
        1. Key Ideas: Main concepts or solutions
        2. Technical Details: Implementation specifics
        3. Impact Analysis: Expected outcomes
        4. Risk Assessment: Potential challenges
        Focus on your expertise area and be specific.
    """
        ),
    ]
)

# Results Synthesis Prompt
synthesis_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "You are an NGT facilitator synthesizing expert ideas."
        ),
        HumanMessagePromptTemplate.from_template(
            """
        Synthesize the following NGT expert ideas into a final recommendation.
        Expert Ideas:
        {responses}
        Context:
        {context}
        Structure your synthesis as:
        1. Executive Summary
        2. Consolidated Ideas
        3. Implementation Plan
        4. Risk Mitigation
        5. Success Metrics
        6. Next Steps
        Focus on actionable outcomes and consensus points.
    """
        ),
    ]
)

aggregation_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """You are an expert insight synthesizer
              specializing in comprehensive analysis."""
        ),
        HumanMessagePromptTemplate.from_template(
            """
                    Synthesize the following expert analyses into a comprehensive report.
                    Expert Analyses:
                    {responses}
                    Context:
                    {context}
                    Follow this structured approach:
                    1. Executive Summary (3-5 key points)
                    2. Expert Consensus Points
                    3. Areas of Disagreement & Resolution
                    4. Unique Insights by Expert
                    5. Integrated Analysis
                    6. Risk Assessment
                    7. Strategic Recommendations
                    Format in markdown and ensure recommendations are specific and actionable.
                """  # noqa: E501
        ),
    ]
)
