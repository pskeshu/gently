"""
Research tools — literature search, strain/gene databases.

These tools let the copilot search external knowledge sources to
inform experimental design. Results are returned as context for the
LLM to interpret, not as final answers.

APIs used:
- PubMed E-utilities (esearch + esummary) for literature
- NCBI Gene database for gene search
- WormBase REST API for strain details per gene
"""

import json
import logging
from typing import Dict, List, Optional

from ...tool_registry import tool, ToolCategory, ToolExample

logger = logging.getLogger(__name__)

# Timeout for external API calls (seconds)
_API_TIMEOUT = 10

# NCBI E-utilities polite-access parameters
_NCBI_TOOL = "gently"
_NCBI_EMAIL = "pskeshu@gmail.com"


def _ncbi_params(**kwargs) -> Dict:
    """Add standard NCBI tool/email to query params."""
    kwargs["tool"] = _NCBI_TOOL
    kwargs["email"] = _NCBI_EMAIL
    kwargs["retmode"] = "json"
    return kwargs


# ---------------------------------------------------------------------------
# PubMed literature search
# ---------------------------------------------------------------------------

async def _pubmed_search(query: str, max_results: int) -> List[Dict]:
    """Search PubMed via E-utilities and return article summaries."""
    import aiohttp

    results: List[Dict] = []

    try:
        async with aiohttp.ClientSession() as session:
            # Step 1: esearch — get PMIDs
            async with session.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params=_ncbi_params(
                    db="pubmed",
                    term=query,
                    retmax=str(max_results),
                ),
                timeout=aiohttp.ClientTimeout(total=_API_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    logger.debug(f"PubMed esearch HTTP {resp.status}")
                    return []
                data = await resp.json()

            id_list = data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return []

            # Step 2: esummary — get article details
            async with session.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                params=_ncbi_params(
                    db="pubmed",
                    id=",".join(id_list),
                ),
                timeout=aiohttp.ClientTimeout(total=_API_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    logger.debug(f"PubMed esummary HTTP {resp.status}")
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
                    first = authors[0].get("name", "")
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
# NCBI Gene search
# ---------------------------------------------------------------------------

async def _ncbi_gene_search(query: str, max_results: int = 5) -> List[Dict]:
    """Search NCBI Gene database for C. elegans genes."""
    import aiohttp

    results: List[Dict] = []

    try:
        async with aiohttp.ClientSession() as session:
            # esearch — find gene IDs
            term = f"{query} AND Caenorhabditis elegans[orgn]"
            async with session.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params=_ncbi_params(
                    db="gene",
                    term=term,
                    retmax=str(max_results),
                ),
                timeout=aiohttp.ClientTimeout(total=_API_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()

            id_list = data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return []

            # esummary — get gene details
            async with session.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                params=_ncbi_params(
                    db="gene",
                    id=",".join(id_list),
                ),
                timeout=aiohttp.ClientTimeout(total=_API_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()

            result_data = data.get("result", {})
            for gid in id_list:
                gene = result_data.get(gid)
                if not gene or not isinstance(gene, dict):
                    continue

                # Only include C. elegans genes
                org = gene.get("organism", {})
                if org.get("taxid") != 6239:
                    continue

                results.append({
                    "gene_id": gid,
                    "name": gene.get("name", ""),
                    "description": gene.get("description", ""),
                    "summary": gene.get("summary", ""),
                    "chromosome": gene.get("chromosome", ""),
                    "aliases": gene.get("otheraliases", ""),
                })

    except Exception as e:
        logger.debug(f"NCBI Gene search failed: {e}")

    return results


# ---------------------------------------------------------------------------
# WormBase strain lookup (by gene ID)
# ---------------------------------------------------------------------------

async def _wormbase_gene_strains(wbgene_id: str) -> List[Dict]:
    """Get strains carrying a gene from WormBase REST API.

    Parameters
    ----------
    wbgene_id : str
        WormBase gene ID, e.g. "WBGene00004295"
    """
    import aiohttp

    results: List[Dict] = []

    try:
        url = f"https://rest.wormbase.org/rest/field/gene/{wbgene_id}/strains"
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                headers={"Accept": "application/json"},
                timeout=aiohttp.ClientTimeout(total=_API_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    logger.debug(f"WormBase strains HTTP {resp.status} for {wbgene_id}")
                    return []
                data = await resp.json()

        strains_data = data.get("strains", {}).get("data", {})
        if not strains_data:
            return []

        # Collect from all categories, noting CGC availability
        for category, strain_list in strains_data.items():
            if not isinstance(strain_list, list):
                continue
            is_cgc = "cgc" in category.lower()
            for entry in strain_list:
                if not isinstance(entry, dict):
                    continue
                results.append({
                    "name": entry.get("label", ""),
                    "genotype": entry.get("genotype", ""),
                    "cgc_available": is_cgc or category == "available_from_cgc",
                    "category": category,
                })

    except Exception as e:
        logger.debug(f"WormBase strain lookup failed: {e}")

    return results


async def _wormbase_gene_id_lookup(gene_name: str) -> Optional[str]:
    """Look up a WormBase gene ID from an NCBI gene name.

    Uses NCBI gene → dbxrefs to find WormBase ID, or falls back to
    the WormBase REST API name endpoint.
    """
    import aiohttp

    try:
        # Try NCBI gene search with exact name match
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params=_ncbi_params(
                    db="gene",
                    term=f"{gene_name}[Gene Name] AND Caenorhabditis elegans[orgn]",
                    retmax="1",
                ),
                timeout=aiohttp.ClientTimeout(total=_API_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()

            id_list = data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return None

            # Get full gene record as XML to find WormBase xref
            # Must use retmode=xml — WBGene IDs don't appear in json
            efetch_params = _ncbi_params(
                db="gene",
                id=id_list[0],
            )
            efetch_params["retmode"] = "xml"
            async with session.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params=efetch_params,
                timeout=aiohttp.ClientTimeout(total=_API_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    return None
                text = await resp.text()

            # Look for WBGene ID in the response text
            import re
            match = re.search(r'(WBGene\d+)', text)
            if match:
                return match.group(1)

    except Exception as e:
        logger.debug(f"WormBase gene ID lookup failed for {gene_name}: {e}")

    return None


# ---------------------------------------------------------------------------
# CGC strain search (HTML scraping fallback)
# ---------------------------------------------------------------------------

async def _cgc_search(query: str, field: str = "strain") -> List[Dict]:
    """Search CGC (Caenorhabditis Genetics Center) strain database.

    Parameters
    ----------
    query : str
        Search term (strain name, gene name, or keyword).
    field : str
        Field to search: "strain", "genotype", "description", "all".
    """
    import aiohttp
    import re

    results: List[Dict] = []

    try:
        url = "https://cgc.umn.edu/strain/search"
        params = {"st1": query, "sf1": field}
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=_API_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    logger.debug(f"CGC search HTTP {resp.status}")
                    return []
                html = await resp.text()

        # Parse strain entries from HTML table rows
        # CGC uses table rows with strain name, species, genotype, description
        # Pattern: look for strain links like /strain/OH904
        strain_pattern = re.compile(
            r'/strain/([A-Z]{1,3}\d+).*?'
            r'<td[^>]*>(.*?)</td>.*?'  # species
            r'<td[^>]*>(.*?)</td>.*?'  # genotype
            r'<td[^>]*>(.*?)</td>',     # description
            re.DOTALL,
        )

        for match in strain_pattern.finditer(html):
            strain_name = match.group(1).strip()
            genotype = re.sub(r'<[^>]+>', '', match.group(3)).strip()
            description = re.sub(r'<[^>]+>', '', match.group(4)).strip()

            if strain_name:
                results.append({
                    "name": strain_name,
                    "genotype": genotype,
                    "description": description[:200] if description else "",
                    "source": "CGC",
                })

        # If regex didn't work, try a simpler approach — find strain names
        if not results:
            simple_pattern = re.compile(r'href="/strain/([A-Z]{1,3}\d+)"')
            seen = set()
            for m in simple_pattern.finditer(html):
                name = m.group(1)
                if name not in seen:
                    seen.add(name)
                    results.append({
                        "name": name,
                        "genotype": "",
                        "description": "",
                        "source": "CGC",
                    })

        # Check for "no results" message
        if "no results for this search" in html.lower():
            return []

    except Exception as e:
        logger.debug(f"CGC search failed: {e}")

    return results[:10]


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

    # Citation-ready block for plan item references
    lines.append("\n---")
    lines.append("**Citation references for plan items** (pass to `references` param):")
    for r in results:
        author = r['authors'] or 'Unknown'
        year = r['year'] or ''
        journal = r['journal'] or ''
        cite = f"{author} ({year}) {r['title'][:80]}. {journal}"
        ref = {
            "source": "pubmed",
            "citation": cite,
            "id": f"PMID:{r['pmid']}",
        }
        lines.append(f"  {json.dumps(ref)}")

    return "\n".join(lines)


@tool(
    name="search_strains",
    description=(
        "Search for C. elegans strains by gene name or keyword. "
        "Searches the NCBI Gene database to find matching genes, then "
        "queries WormBase for strains carrying those genes, including "
        "CGC availability and genotype. Also works as a general gene "
        "search to find gene descriptions and aliases."
    ),
    category=ToolCategory.UTILITY,
    examples=[
        ToolExample(
            user_query="What GFP reporters exist for nerve ring neurons?",
            tool_input={
                "query": "rab-3",
                "organism": "celegans",
            },
        ),
        ToolExample(
            user_query="Find strains for unc-6",
            tool_input={
                "query": "unc-6",
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
    """Search for strains and genes via NCBI Gene + WormBase REST API.

    Falls back to Claude's built-in knowledge if APIs are unavailable.
    """
    if organism != "celegans":
        return (
            f"[Strain search for: '{query}' ({organism})]\n\n"
            f"Strain database search is currently only supported for C. elegans. "
            f"I'll use my knowledge of available {organism} strains to suggest options."
        )

    import re as _re

    # Detect if query looks like a strain name (e.g. OH904, N2, CB1370)
    is_strain_query = bool(_re.match(r'^[A-Z]{1,3}\d+$', query.strip()))

    # Step 0: If it looks like a strain name, search CGC directly
    cgc_results = []
    if is_strain_query:
        cgc_results = await _cgc_search(query, field="strain")
    else:
        # Also try CGC genotype search for gene-like queries
        cgc_results = await _cgc_search(query, field="genotype")

    # Step 1: Search NCBI Gene database
    genes = await _ncbi_gene_search(query, max_results=5)

    if not genes and not cgc_results:
        return (
            f"[Gene/strain search for: '{query}']\n\n"
            f"No results from NCBI Gene or CGC databases. "
            f"I'll use my knowledge of available strains and genes "
            f"to suggest options. If you know what strains your lab has, please "
            f"tell me and I'll incorporate that into the plan."
        )

    lines = []

    # Show CGC results first if we have them
    if cgc_results:
        lines.append(f"CGC (Caenorhabditis Genetics Center) results for: '{query}'\n")
        for i, s in enumerate(cgc_results, 1):
            lines.append(f"{i}. **{s['name']}** [CGC — available to order]")
            if s["genotype"]:
                lines.append(f"   Genotype: {s['genotype']}")
            if s["description"]:
                lines.append(f"   {s['description']}")
            lines.append("")
        if genes:
            lines.append("---\n")

    if genes:
        lines.append(f"NCBI Gene results for: '{query}'\n")

    for i, gene in enumerate(genes, 1):
        lines.append(f"**{i}. {gene['name']}** (NCBI Gene ID: {gene['gene_id']})")
        if gene["description"]:
            lines.append(f"   {gene['description']}")
        if gene["chromosome"]:
            lines.append(f"   Chromosome: {gene['chromosome']}")
        if gene["aliases"]:
            lines.append(f"   Aliases: {gene['aliases']}")

        # Step 2: Try to get WormBase strains for the top gene(s)
        if i <= 2:  # Only look up strains for top 2 to keep latency reasonable
            wbgene_id = await _wormbase_gene_id_lookup(gene["name"])
            if wbgene_id:
                strains = await _wormbase_gene_strains(wbgene_id)
                if strains:
                    lines.append(f"   **Strains ({len(strains)} found):**")
                    # Show up to 6, prioritising CGC-available
                    sorted_strains = sorted(
                        strains, key=lambda s: (not s["cgc_available"], s["name"]),
                    )
                    for s in sorted_strains[:6]:
                        cgc_tag = " [CGC]" if s["cgc_available"] else ""
                        geno = f" — {s['genotype']}" if s["genotype"] else ""
                        lines.append(f"   - {s['name']}{cgc_tag}{geno}")
                    if len(strains) > 6:
                        lines.append(f"   - ... and {len(strains) - 6} more")
                else:
                    lines.append("   No strains found in WormBase for this gene.")
            else:
                lines.append("   (WormBase ID not resolved — strain lookup skipped)")

        if gene["summary"]:
            # Truncate long summaries
            summary = gene["summary"]
            if len(summary) > 200:
                summary = summary[:200] + "..."
            lines.append(f"   Summary: {summary}")

        lines.append("")

    lines.append(
        "I can provide more details about specific genes or strains, "
        "including expression patterns, phenotypes, and reporter availability."
    )

    # Citation-ready block for plan item references
    citation_refs = []
    if cgc_results:
        for s in cgc_results:
            citation_refs.append({
                "source": "cgc",
                "citation": f"{s['name']} available from CGC",
                "id": f"CGC:{s['name']}",
            })
    for gene in genes:
        citation_refs.append({
            "source": "ncbi_gene",
            "citation": f"{gene['name']} — {gene['description']}",
            "id": f"GeneID:{gene['gene_id']}",
        })
    if citation_refs:
        lines.append("\n---")
        lines.append("**Citation references for plan items** (pass to `references` param):")
        for ref in citation_refs:
            lines.append(f"  {json.dumps(ref)}")

    return "\n".join(lines)
