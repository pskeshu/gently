"""
Research tools — literature search, strain databases, paper ingestion.

These tools let the copilot search external knowledge sources to
inform experimental design.
"""

from typing import Dict, Optional

from ...tool_registry import tool, ToolCategory, ToolExample


@tool(
    name="search_literature",
    description=(
        "Search scientific literature (PubMed, Google Scholar) for relevant "
        "papers. Returns titles, authors, and key findings. Use this to find "
        "published methods, strains, imaging approaches, and prior work."
    ),
    category=ToolCategory.UTILITY,
    examples=[
        ToolExample(
            user_query="Find papers on nerve ring formation live imaging",
            tool_input={
                "query": "C. elegans nerve ring formation live imaging",
            },
        ),
    ],
)
async def search_literature(
    query: str,
    max_results: int = 5,
    context: Dict = None,
) -> str:
    """Search scientific literature for relevant papers.

    Note: This is a placeholder that uses Claude's training knowledge.
    A future version will integrate with PubMed/Google Scholar APIs.
    """
    # For now, return a note that this uses Claude's built-in knowledge.
    # The copilot will use its training data to reason about literature.
    # Future: integrate with PubMed E-utilities API or Google Scholar.
    return (
        f"[Literature search for: '{query}']\n\n"
        f"Note: Direct PubMed/Scholar search is not yet connected. "
        f"I'll use my knowledge of the published literature to inform "
        f"the experimental design. If you have specific papers in mind, "
        f"you can share them and I can read the PDFs.\n\n"
        f"Based on my training knowledge, I'll incorporate relevant "
        f"published methods, strains, and findings into the plan."
    )


@tool(
    name="search_strains",
    description=(
        "Search for available strains and reporters in strain databases "
        "(WormBase, CGC, FlyBase). Returns strain names, genotypes, and "
        "reporter information."
    ),
    category=ToolCategory.UTILITY,
    examples=[
        ToolExample(
            user_query="What GFP reporters exist for nerve ring neurons?",
            tool_input={
                "query": "pan-neuronal GFP reporter",
                "organism": "celegans",
            },
        ),
    ],
)
async def search_strains(
    query: str,
    organism: str = "celegans",
    context: Dict = None,
) -> str:
    """Search strain databases for available strains and reporters.

    Note: This is a placeholder that uses Claude's training knowledge.
    A future version will integrate with WormBase/CGC APIs.
    """
    return (
        f"[Strain search for: '{query}' ({organism})]\n\n"
        f"Note: Direct WormBase/CGC search is not yet connected. "
        f"I'll use my knowledge of available strains to suggest options. "
        f"If you know what strains your lab has, please tell me and "
        f"I'll incorporate that into the plan.\n\n"
        f"Based on my training knowledge, I'll suggest specific strains "
        f"with real CGC names and genotypes."
    )
