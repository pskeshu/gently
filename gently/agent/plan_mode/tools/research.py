"""
Research tools — literature search, strain databases, paper ingestion.

These tools let the copilot search external knowledge sources to
inform experimental design. Results are returned as context for the
LLM to interpret, not as final answers.
"""

import logging
from typing import Dict, List, Optional
from xml.etree import ElementTree

from ...tool_registry import tool, ToolCategory, ToolExample

logger = logging.getLogger(__name__)

# Timeout for external API calls (seconds)
_API_TIMEOUT = 10


# ---------------------------------------------------------------------------
# PubMed helpers
# ---------------------------------------------------------------------------

async def _pubmed_search(query: str, max_results: int) -> List[Dict]:
    """Search PubMed via E-utilities and return article summaries."""
    import aiohttp

    results: List[Dict] = []

    try:
        async with aiohttp.ClientSession() as session:
            # Step 1: esearch to get PMIDs
            search_url = (
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            )
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": str(max_results),
                "retmode": "json",
            }
            async with session.get(
                search_url, params=search_params, timeout=aiohttp.ClientTimeout(total=_API_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()

            id_list = data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return []

            # Step 2: esummary to get article details
            summary_url = (
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            )
            summary_params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "json",
            }
            async with session.get(
                summary_url, params=summary_params, timeout=aiohttp.ClientTimeout(total=_API_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()

            result_data = data.get("result", {})
            for pmid in id_list:
                article = result_data.get(pmid)
                if not article or not isinstance(article, dict):
                    continue

                # Extract author list
                authors = article.get("authors", [])
                author_str = ""
                if authors:
                    first = authors[0].get("name", "") if authors else ""
                    if len(authors) > 1:
                        author_str = f"{first} et al."
                    else:
                        author_str = first

                results.append({
                    "pmid": pmid,
                    "title": article.get("title", ""),
                    "authors": author_str,
                    "journal": article.get("fulljournalname", article.get("source", "")),
                    "year": article.get("pubdate", "")[:4],
                    "doi": next(
                        (
                            aid.get("value", "")
                            for aid in article.get("articleids", [])
                            if aid.get("idtype") == "doi"
                        ),
                        "",
                    ),
                })

    except Exception as e:
        logger.debug(f"PubMed search failed: {e}")

    return results


# ---------------------------------------------------------------------------
# WormBase helpers
# ---------------------------------------------------------------------------

async def _wormbase_search(query: str) -> List[Dict]:
    """Search WormBase for strains matching the query."""
    import aiohttp

    results: List[Dict] = []

    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://wormbase.org/search/strain/{query}"
            headers = {"Accept": "application/json"}
            async with session.get(
                url, headers=headers, timeout=aiohttp.ClientTimeout(total=_API_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()

            # WormBase search returns results in data.results
            search_results = data.get("results", [])
            if not search_results:
                # Try alternate structure
                search_results = data.get("data", {}).get("results", [])

            for entry in search_results[:8]:
                if isinstance(entry, dict):
                    name = entry.get("name", entry.get("label", ""))
                    genotype = entry.get("genotype", "")
                    results.append({
                        "name": str(name),
                        "genotype": str(genotype),
                    })
                elif isinstance(entry, str):
                    results.append({"name": entry, "genotype": ""})

    except Exception as e:
        logger.debug(f"WormBase search failed: {e}")

    return results


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool(
    name="search_literature",
    description=(
        "Search scientific literature (PubMed) for relevant papers. "
        "Returns titles, authors, journal, year, and DOI. Use this to find "
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
    """Search PubMed for relevant papers.

    Falls back to Claude's built-in knowledge if the API is unavailable.
    """
    if max_results < 1:
        max_results = 5
    if max_results > 8:
        max_results = 8

    results = await _pubmed_search(query, max_results)

    if not results:
        return (
            f"[Literature search for: '{query}']\n\n"
            f"PubMed search returned no results or is currently unavailable. "
            f"I'll use my knowledge of the published literature to inform "
            f"the experimental design. If you have specific papers in mind, "
            f"you can share them and I can read the PDFs."
        )

    lines = [f"PubMed results for: '{query}' ({len(results)} found)\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. **{r['title']}**")
        parts = []
        if r["authors"]:
            parts.append(r["authors"])
        if r["journal"]:
            parts.append(f"*{r['journal']}*")
        if r["year"]:
            parts.append(f"({r['year']})")
        if parts:
            lines.append(f"   {' '.join(parts)}")
        if r["doi"]:
            lines.append(f"   DOI: {r['doi']}")
        lines.append(f"   PMID: {r['pmid']}")
        lines.append("")

    lines.append(
        "Use these results as context for experimental design. "
        "I can discuss any of these papers in more detail."
    )
    return "\n".join(lines)


@tool(
    name="search_strains",
    description=(
        "Search for available strains and reporters in strain databases "
        "(WormBase, CGC). Returns strain names, genotypes, and availability. "
        "Results supplement Claude's knowledge of available strains."
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
    """Search WormBase for available strains and reporters.

    Falls back to Claude's built-in knowledge if the API is unavailable.
    """
    if organism != "celegans":
        return (
            f"[Strain search for: '{query}' ({organism})]\n\n"
            f"Strain database search is currently only supported for C. elegans. "
            f"I'll use my knowledge of available {organism} strains to suggest options."
        )

    results = await _wormbase_search(query)

    if not results:
        return (
            f"[Strain search for: '{query}']\n\n"
            f"WormBase search returned no results or is currently unavailable. "
            f"I'll use my knowledge of available strains to suggest options. "
            f"If you know what strains your lab has, please tell me and "
            f"I'll incorporate that into the plan."
        )

    lines = [f"WormBase results for: '{query}' ({len(results)} found)\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. **{r['name']}**")
        if r.get("genotype"):
            lines.append(f"   Genotype: {r['genotype']}")
        lines.append("")

    lines.append(
        "These are database matches. I can provide more details about "
        "specific strains, including reporter expression patterns and CGC availability."
    )
    return "\n".join(lines)
