"""
Research tools — literature search, strain/gene databases.

These tools let the agent search external knowledge sources to
inform experimental design. Results are returned as context for the
LLM to interpret, not as final answers.

APIs used:
- PubMed E-utilities (esearch + esummary) for literature
- NCBI Gene database for gene search
- WormBase REST API for strain details per gene
"""

import json
import logging
import ssl

from ....settings import settings
from ...tools.registry import ToolCategory, ToolExample, tool

logger = logging.getLogger(__name__)

# Timeout for external API calls (seconds)
_API_TIMEOUT = settings.timeouts.api_call

# NCBI E-utilities polite-access parameters
_NCBI_TOOL = settings.api.ncbi_tool
_NCBI_EMAIL = settings.api.ncbi_email


def _ssl_context() -> ssl.SSLContext:
    """Create an SSL context using certifi's CA bundle.

    The system cert file (C:\\Program Files\\Common Files\\SSL\\cert.pem)
    doesn't exist on this machine, so we use certifi explicitly.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _ncbi_params(**kwargs) -> dict:
    """Add standard NCBI tool/email to query params."""
    kwargs["tool"] = _NCBI_TOOL
    kwargs["email"] = _NCBI_EMAIL
    kwargs["retmode"] = "json"
    return kwargs


def _http_session():
    """Create an aiohttp ClientSession with explicit SSL certs."""
    import aiohttp

    connector = aiohttp.TCPConnector(ssl=_ssl_context())
    return aiohttp.ClientSession(connector=connector)


# ---------------------------------------------------------------------------
# PubMed literature search
# ---------------------------------------------------------------------------


async def _pubmed_search(query: str, max_results: int) -> list[dict]:
    """Search PubMed via E-utilities and return article summaries."""
    import aiohttp

    results: list[dict] = []

    try:
        async with _http_session() as session:
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
                    logger.warning(f"PubMed esearch HTTP {resp.status}")
                    return []
                # content_type=None avoids ContentTypeError when NCBI
                # returns text/plain or other non-json content types
                data = await resp.json(content_type=None)

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
                    logger.warning(f"PubMed esummary HTTP {resp.status}")
                    return []
                data = await resp.json(content_type=None)

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

                results.append(
                    {
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
                    }
                )

    except Exception as e:
        logger.warning(f"PubMed search failed: {e}", exc_info=True)

    return results


# ---------------------------------------------------------------------------
# NCBI Gene search
# ---------------------------------------------------------------------------


async def _ncbi_gene_search(query: str, max_results: int = 5) -> list[dict]:
    """Search NCBI Gene database for C. elegans genes."""
    import aiohttp

    results: list[dict] = []

    try:
        async with _http_session() as session:
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
                    logger.warning(f"NCBI Gene esearch HTTP {resp.status}")
                    return []
                data = await resp.json(content_type=None)

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
                    logger.warning(f"NCBI Gene esummary HTTP {resp.status}")
                    return []
                data = await resp.json(content_type=None)

            result_data = data.get("result", {})
            for gid in id_list:
                gene = result_data.get(gid)
                if not gene or not isinstance(gene, dict):
                    continue

                # Only include C. elegans genes
                org = gene.get("organism", {})
                if org.get("taxid") != 6239:
                    continue

                results.append(
                    {
                        "gene_id": gid,
                        "name": gene.get("name", ""),
                        "description": gene.get("description", ""),
                        "summary": gene.get("summary", ""),
                        "chromosome": gene.get("chromosome", ""),
                        "aliases": gene.get("otheraliases", ""),
                    }
                )

    except Exception as e:
        logger.warning(f"NCBI Gene search failed: {e}", exc_info=True)

    return results


# ---------------------------------------------------------------------------
# WormBase strain lookup (by gene ID)
# ---------------------------------------------------------------------------


async def _wormbase_gene_strains(wbgene_id: str) -> list[dict]:
    """Get strains carrying a gene from WormBase REST API.

    Parameters
    ----------
    wbgene_id : str
        WormBase gene ID, e.g. "WBGene00004295"
    """
    import aiohttp

    results: list[dict] = []

    try:
        url = f"https://rest.wormbase.org/rest/field/gene/{wbgene_id}/strains"
        async with _http_session() as session:
            async with session.get(
                url,
                headers={"Accept": "application/json"},
                timeout=aiohttp.ClientTimeout(total=_API_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"WormBase strains HTTP {resp.status} for {wbgene_id}")
                    return []
                data = await resp.json(content_type=None)

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
                results.append(
                    {
                        "name": entry.get("label", ""),
                        "genotype": entry.get("genotype", ""),
                        "cgc_available": is_cgc or category == "available_from_cgc",
                        "category": category,
                    }
                )

    except Exception as e:
        logger.warning(f"WormBase strain lookup failed: {e}", exc_info=True)

    return results


async def _wormbase_gene_id_lookup(gene_name: str) -> str | None:
    """Look up a WormBase gene ID from an NCBI gene name.

    Uses NCBI gene → dbxrefs to find WormBase ID, or falls back to
    the WormBase REST API name endpoint.
    """
    import aiohttp

    try:
        # Try NCBI gene search with exact name match
        async with _http_session() as session:
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
                data = await resp.json(content_type=None)

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

            match = re.search(r"(WBGene\d+)", text)
            if match:
                return match.group(1)

    except Exception as e:
        logger.warning(f"WormBase gene ID lookup failed for {gene_name}: {e}")

    return None


# ---------------------------------------------------------------------------
# CGC strain search (HTML scraping fallback)
# ---------------------------------------------------------------------------


async def _cgc_search(query: str, field: str = "strain") -> list[dict]:
    """Search CGC (Caenorhabditis Genetics Center) strain database.

    Parameters
    ----------
    query : str
        Search term (strain name, gene name, or keyword).
    field : str
        Field to search: "strain", "genotype", "description", "all".
    """
    import re

    import aiohttp

    results: list[dict] = []

    try:
        url = "https://cgc.umn.edu/strain/search"
        params = {"st1": query, "sf1": field}
        async with _http_session() as session:
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
            r"/strain/([A-Z]{1,3}\d+).*?"
            r"<td[^>]*>(.*?)</td>.*?"  # species
            r"<td[^>]*>(.*?)</td>.*?"  # genotype
            r"<td[^>]*>(.*?)</td>",  # description
            re.DOTALL,
        )

        for match in strain_pattern.finditer(html):
            strain_name = match.group(1).strip()
            genotype = re.sub(r"<[^>]+>", "", match.group(3)).strip()
            description = re.sub(r"<[^>]+>", "", match.group(4)).strip()

            if strain_name:
                results.append(
                    {
                        "name": strain_name,
                        "genotype": genotype,
                        "description": description[:200] if description else "",
                        "source": "CGC",
                    }
                )

        # If regex didn't work, try a simpler approach — find strain names
        if not results:
            simple_pattern = re.compile(r'href="/strain/([A-Z]{1,3}\d+)"')
            seen = set()
            for m in simple_pattern.finditer(html):
                name = m.group(1)
                if name not in seen:
                    seen.add(name)
                    results.append(
                        {
                            "name": name,
                            "genotype": "",
                            "description": "",
                            "source": "CGC",
                        }
                    )

        # Check for "no results" message
        if "no results for this search" in html.lower():
            return []

    except Exception as e:
        logger.warning(f"CGC search failed: {e}", exc_info=True)

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
    context: dict | None = None,
) -> str:
    """Search PubMed for relevant papers.

    If the original query is too specific (zero hits), progressively
    simplifies by dropping terms until results are found.
    """
    if max_results < 1:
        max_results = 5
    if max_results > 8:
        max_results = 8

    results = await _pubmed_search(query, max_results)
    used_query = query

    # If no results, try progressively simpler queries
    if not results:
        words = query.split()
        if len(words) > 3:
            # Strategy 1: keep first ~60% of terms (drop trailing specifics)
            shorter = " ".join(words[: max(3, len(words) * 2 // 3)])
            results = await _pubmed_search(shorter, max_results)
            if results:
                used_query = shorter

        if not results and len(words) > 4:
            # Strategy 2: keep only the core noun phrases (drop adjectives/filler)
            # Remove common filler words
            stopwords = {
                "and",
                "or",
                "the",
                "of",
                "in",
                "for",
                "with",
                "a",
                "an",
                "using",
                "based",
                "via",
                "during",
                "after",
                "before",
            }
            core = [w for w in words if w.lower() not in stopwords]
            if len(core) > 3:
                core = core[:4]
            shorter = " ".join(core)
            results = await _pubmed_search(shorter, max_results)
            if results:
                used_query = shorter

        if not results and len(words) > 2:
            # Strategy 3: just the first 3 words
            shorter = " ".join(words[:3])
            results = await _pubmed_search(shorter, max_results)
            if results:
                used_query = shorter

    if not results:
        return (
            f"[Literature search for: '{query}']\n\n"
            f"No PubMed results found, even with simplified queries. "
            f"Try shorter, more specific terms (e.g. 'C. elegans light sheet' "
            f"instead of 'C. elegans light sheet fluorescence long-term imaging'). "
            f"You can also use read_paper with a specific PMID or DOI."
        )

    header = f"PubMed results for: '{query}' ({len(results)} found)"
    if used_query != query:
        header += f"\n*(original query too specific — matched on: '{used_query}')*"
    lines = [header + "\n"]
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
        author = r["authors"] or "Unknown"
        year = r["year"] or ""
        journal = r["journal"] or ""
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
    context: dict | None = None,
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
    is_strain_query = bool(_re.match(r"^[A-Z]{1,3}\d+$", query.strip()))

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
                        strains,
                        key=lambda s: (not s["cgc_available"], s["name"]),
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
            citation_refs.append(
                {
                    "source": "cgc",
                    "citation": f"{s['name']} available from CGC",
                    "id": f"CGC:{s['name']}",
                }
            )
    for gene in genes:
        citation_refs.append(
            {
                "source": "ncbi_gene",
                "citation": f"{gene['name']} — {gene['description']}",
                "id": f"GeneID:{gene['gene_id']}",
            }
        )
    if citation_refs:
        lines.append("\n---")
        lines.append("**Citation references for plan items** (pass to `references` param):")
        for ref in citation_refs:
            lines.append(f"  {json.dumps(ref)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Paper reading — full text retrieval
# ---------------------------------------------------------------------------


async def _pmid_to_pmcid(pmid: str) -> str | None:
    """Convert a PMID to a PMCID via the NCBI ID converter."""
    import aiohttp

    try:
        async with _http_session() as session:
            async with session.get(
                "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/",
                params={
                    "ids": pmid,
                    "format": "json",
                    "tool": _NCBI_TOOL,
                    "email": _NCBI_EMAIL,
                },
                timeout=aiohttp.ClientTimeout(total=_API_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json(content_type=None)

        records = data.get("records", [])
        if records and records[0].get("pmcid"):
            return records[0]["pmcid"]
    except Exception as e:
        logger.warning(f"PMID→PMCID conversion failed for {pmid}: {e}")

    return None


async def _fetch_pmc_fulltext(pmcid: str) -> str | None:
    """Fetch full text from PubMed Central as XML, parse into sections."""
    import xml.etree.ElementTree as ET

    import aiohttp

    try:
        # Strip "PMC" prefix for efetch — it wants just the number
        pmc_num = pmcid.replace("PMC", "")

        async with _http_session() as session:
            params = _ncbi_params(db="pmc", id=pmc_num)
            params["rettype"] = "xml"
            params.pop("retmode", None)  # PMC XML doesn't use retmode=json

            async with session.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params=params,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"PMC efetch HTTP {resp.status} for {pmcid}")
                    return None
                xml_text = await resp.text()

        root = ET.fromstring(xml_text)
        return _parse_pmc_xml(root)

    except Exception as e:
        logger.warning(f"PMC full text fetch failed for {pmcid}: {e}")
        return None


def _parse_pmc_xml(root) -> str:
    """Extract structured text from PMC XML (JATS format)."""
    sections = []

    # Article metadata
    article = root.find(".//article-meta")
    if article is not None:
        title_el = article.find(".//article-title")
        if title_el is not None:
            sections.append(f"# {_xml_text(title_el)}\n")

        # Authors
        authors = []
        for contrib in article.findall(".//contrib[@contrib-type='author']"):
            surname = contrib.findtext("name/surname", "")
            given = contrib.findtext("name/given-names", "")
            if surname:
                authors.append(f"{given} {surname}".strip())
        if authors:
            sections.append(f"**Authors:** {', '.join(authors)}\n")

        # Abstract
        abstract = article.find(".//abstract")
        if abstract is not None:
            sections.append("## Abstract\n")
            sections.append(_xml_text(abstract) + "\n")

    # Body sections
    body = root.find(".//body")
    if body is not None:
        for sec in body.findall(".//sec"):
            title_el = sec.find("title")
            title = _xml_text(title_el) if title_el is not None else ""
            if title:
                # Determine heading level by nesting depth
                depth = 0
                parent = sec
                while parent is not None:
                    parent = _find_parent(body, parent)
                    if parent is not None and parent.tag == "sec":
                        depth += 1
                level = min(depth + 2, 4)  # ##, ###, ####
                sections.append(f"{'#' * level} {title}\n")

            # Collect paragraph text (skip nested sec)
            for child in sec:
                if child.tag == "p":
                    sections.append(_xml_text(child) + "\n")
                elif child.tag == "table-wrap":
                    caption = child.find(".//caption")
                    if caption is not None:
                        sections.append(f"*Table: {_xml_text(caption)}*\n")
                elif child.tag == "fig":
                    caption = child.find(".//caption")
                    if caption is not None:
                        label = child.findtext("label", "Figure")
                        sections.append(f"*{label}: {_xml_text(caption)}*\n")

    text = "\n".join(sections)

    # Truncate if very long (keep under ~15k chars for context window)
    if len(text) > 15000:
        text = text[:15000] + "\n\n[... truncated — paper continues ...]"

    return text


def _xml_text(element) -> str:
    """Extract all text from an XML element, including nested elements."""
    if element is None:
        return ""
    parts = []
    parts.append(element.text or "")
    for child in element:
        parts.append(_xml_text(child))
        parts.append(child.tail or "")
    return "".join(parts).strip()


def _find_parent(root, target):
    """Find the parent of an element in an XML tree."""
    for parent in root.iter():
        for child in parent:
            if child is target:
                return parent
    return None


async def _unpaywall_lookup(doi: str) -> str | None:
    """Find an open access full text URL via Unpaywall."""
    import aiohttp

    try:
        async with _http_session() as session:
            async with session.get(
                f"https://api.unpaywall.org/v2/{doi}",
                params={"email": _NCBI_EMAIL},
                timeout=aiohttp.ClientTimeout(total=_API_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json(content_type=None)

        best = data.get("best_oa_location")
        if best:
            return best.get("url_for_pdf") or best.get("url_for_landing_page")

    except Exception as e:
        logger.warning(f"Unpaywall lookup failed for {doi}: {e}")

    return None


async def _fetch_url_text(url: str) -> str | None:
    """Fetch a URL and extract text content (HTML → plain text)."""
    import re as _re

    import aiohttp

    try:
        async with _http_session() as session:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=20),
                headers={"User-Agent": "gently/1.0 (microscopy agent; mailto:" + _NCBI_EMAIL + ")"},
            ) as resp:
                if resp.status != 200:
                    return None
                content_type = resp.content_type or ""

                if "pdf" in content_type:
                    # Can't parse PDF from URL without pymupdf
                    return None

                html = await resp.text()

        # Simple HTML → text: strip tags, collapse whitespace
        text = _re.sub(r"<script[^>]*>.*?</script>", "", html, flags=_re.DOTALL)
        text = _re.sub(r"<style[^>]*>.*?</style>", "", html, flags=_re.DOTALL)
        text = _re.sub(r"<[^>]+>", " ", text)
        text = _re.sub(r"\s+", " ", text).strip()

        if len(text) > 15000:
            text = text[:15000] + "\n\n[... truncated ...]"

        return text if len(text) > 200 else None

    except Exception as e:
        logger.warning(f"URL fetch failed for {url}: {e}")
        return None


def _read_pdf_file(path: str) -> str | None:
    """Extract text from a local PDF file using pymupdf if available."""
    import os

    if not os.path.isfile(path):
        return None

    try:
        import fitz  # pymupdf

        doc = fitz.open(path)
        pages = []
        for page in doc:
            pages.append(page.get_text())
        text = "\n\n".join(pages)
        doc.close()

        if len(text) > 15000:
            text = text[:15000] + "\n\n[... truncated — PDF continues ...]"
        return text if text.strip() else None

    except ImportError:
        logger.info(
            "pymupdf not installed — cannot extract PDF text. Install with: pip install pymupdf"
        )
        return None
    except Exception as e:
        logger.warning(f"PDF extraction failed for {path}: {e}")
        return None


async def _pubmed_abstract(pmid: str) -> dict | None:
    """Fetch article metadata + abstract from PubMed."""
    import aiohttp

    try:
        async with _http_session() as session:
            params = _ncbi_params(db="pubmed", id=pmid)
            params["rettype"] = "abstract"
            params["retmode"] = "xml"

            async with session.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params=params,
                timeout=aiohttp.ClientTimeout(total=_API_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    return None
                xml_text = await resp.text()

        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml_text)

        article = root.find(".//PubmedArticle")
        if article is None:
            return None

        title = article.findtext(".//ArticleTitle", "")

        # Authors
        authors = []
        for author in article.findall(".//Author"):
            last = author.findtext("LastName", "")
            first = author.findtext("ForeName", "")
            if last:
                authors.append(f"{first} {last}".strip())

        # Abstract
        abstract_parts = []
        for abs_text in article.findall(".//AbstractText"):
            label = abs_text.get("Label", "")
            text = _xml_text(abs_text)
            if label:
                abstract_parts.append(f"**{label}:** {text}")
            else:
                abstract_parts.append(text)

        # DOI
        doi = ""
        for aid in article.findall(".//ArticleId"):
            if aid.get("IdType") == "doi":
                doi = aid.text or ""

        journal = article.findtext(".//Journal/Title", "")
        year = article.findtext(".//PubDate/Year", "")

        return {
            "pmid": pmid,
            "title": title,
            "authors": ", ".join(authors),
            "journal": journal,
            "year": year,
            "doi": doi,
            "abstract": "\n\n".join(abstract_parts),
        }

    except Exception as e:
        logger.warning(f"PubMed abstract fetch failed for {pmid}: {e}")
        return None


async def _resolve_reference(reference: str) -> dict:
    """Parse a reference string and determine what kind of input it is.

    Returns a dict with keys: type, pmid, doi, url, path, query
    """
    import os
    import re

    ref = reference.strip()
    result = {"type": "unknown", "raw": ref}

    # PMID
    m = re.match(r"^(?:PMID[:\s]*)?(\d{6,9})$", ref, re.IGNORECASE)
    if m:
        result["type"] = "pmid"
        result["pmid"] = m.group(1)
        return result

    # DOI
    m = re.search(r"(10\.\d{4,}/[^\s]+)", ref)
    if m:
        result["type"] = "doi"
        result["doi"] = m.group(1).rstrip(".,;)")
        return result

    # URL (PubMed, PMC, or other)
    if ref.startswith(("http://", "https://", "www.")):
        result["type"] = "url"
        result["url"] = ref if ref.startswith("http") else "https://" + ref

        # Extract PMID from PubMed URLs
        m = re.search(r"pubmed\.ncbi.*?/(\d{6,9})", ref)
        if m:
            result["type"] = "pmid"
            result["pmid"] = m.group(1)

        # Extract PMCID from PMC URLs
        m = re.search(r"/pmc/articles/(PMC\d+)", ref)
        if m:
            result["type"] = "pmcid"
            result["pmcid"] = m.group(1)

        return result

    # File path (PDF)
    if os.path.isfile(ref) or ref.lower().endswith(".pdf"):
        result["type"] = "file"
        result["path"] = ref
        return result

    # Citation / search query
    result["type"] = "search"
    result["query"] = ref
    return result


async def _search_pmid(query: str) -> str | None:
    """Search PubMed for a citation string and return the best PMID.

    Tries multiple query strategies to handle imprecise citations like
    "Rapti et al 2019 nerve ring" (actual paper is Rapti 2017, Science).
    """
    import re as _re

    strategies = []

    # Detect "Author et al YEAR topic" pattern
    m = _re.match(
        r"^([A-Z][a-z]+)\s+(?:et\s+al\.?\s+)?(\d{4})?\s*(.*)?$",
        query.strip(),
    )
    if m:
        author = m.group(1)
        year = m.group(2)
        topic = (m.group(3) or "").strip()

        # Fix common organism names in topic
        topic_fixed = _re.sub(
            r"\bC\.?\s*elegans\b",
            '"Caenorhabditis elegans"',
            topic,
            flags=_re.IGNORECASE,
        )

        # Strategy 1: author + organism MeSH + quoted topic (most specific)
        if topic_fixed:
            strategies.append(
                f'{author}[author] AND "Caenorhabditis elegans"[Mesh] AND "{topic_fixed}"'
            )

        # Strategy 2: author + organism MeSH (no topic — topic may not be in title)
        strategies.append(f'{author}[author] AND "Caenorhabditis elegans"[Mesh]')

        # Strategy 3: author + year + topic (exact year, may be wrong)
        if year and topic_fixed:
            strategies.append(f"{author}[author] AND {year}[pdat] AND {topic_fixed}")

        # Strategy 4: author + year only
        if year:
            strategies.append(f"{author}[author] AND {year}[pdat]")

    # Strategy 5: original query with organism name fix
    fixed = _re.sub(
        r"\bC\.?\s*elegans\b",
        '"Caenorhabditis elegans"',
        query,
        flags=_re.IGNORECASE,
    )
    if fixed != query:
        strategies.append(fixed)

    # Strategy 6: original query as-is
    strategies.append(query)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in strategies:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    # Try each strategy
    import aiohttp

    for attempt in unique:
        try:
            async with _http_session() as session:
                async with session.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                    params=_ncbi_params(db="pubmed", term=attempt, retmax="1"),
                    timeout=aiohttp.ClientTimeout(total=_API_TIMEOUT),
                ) as resp:
                    if resp.status != 200:
                        continue
                    data = await resp.json(content_type=None)

            ids = data.get("esearchresult", {}).get("idlist", [])
            if ids:
                logger.debug(f"PubMed search hit with: {attempt!r} -> {ids[0]}")
                return ids[0]

        except Exception as e:
            logger.warning(f"PubMed search failed for '{attempt}': {e}")

    return None


async def _doi_to_pmid(doi: str) -> str | None:
    """Resolve a DOI to a PMID via PubMed search."""
    return await _search_pmid(f"{doi}[doi]")


@tool(
    name="read_paper",
    description=(
        "Read a scientific paper given a PMID, DOI, URL, file path, or "
        "citation text (e.g. 'Rapti et al 2019'). Retrieves full text "
        "from PubMed Central when available, checks Unpaywall for open "
        "access versions, reads local PDFs, or falls back to the abstract. "
        "Returns structured text for the agent to analyze."
    ),
    category=ToolCategory.UTILITY,
    examples=[
        ToolExample(
            user_query="Read the Rapti et al paper on nerve ring assembly",
            tool_input={"reference": "Rapti et al 2019 nerve ring"},
        ),
        ToolExample(
            user_query="What does PMID 31537803 say about axon guidance?",
            tool_input={"reference": "31537803"},
        ),
        ToolExample(
            user_query="Read this paper: 10.1016/j.cell.2019.08.023",
            tool_input={"reference": "10.1016/j.cell.2019.08.023"},
        ),
    ],
)
async def read_paper(
    reference: str,
    context: dict | None = None,
) -> str:
    """Read a scientific paper and return its content.

    Tries multiple sources in order:
    1. PubMed Central full text (open access)
    2. Unpaywall OA version
    3. Local PDF file
    4. URL fetch
    5. PubMed abstract (fallback)
    """
    parsed = await _resolve_reference(reference)
    ref_type = parsed["type"]

    pmid = parsed.get("pmid")
    doi = parsed.get("doi")
    pmcid = parsed.get("pmcid")
    url = parsed.get("url")
    path = parsed.get("path")

    status_lines = []  # Track what we tried

    # --- Step 1: Resolve to PMID if we don't have one ---

    if ref_type == "search":
        query = parsed["query"]
        status_lines.append(f"Searching PubMed for: '{query}'")
        pmid = await _search_pmid(query)
        if pmid:
            status_lines.append(f"Found PMID: {pmid}")
        else:
            return (
                f"[Paper lookup: '{reference}']\n\n"
                f"Could not find a matching paper on PubMed for '{query}'. "
                f"Try providing a PMID, DOI, or more specific citation."
            )

    if ref_type == "doi" and not pmid and doi:
        status_lines.append(f"Resolving DOI: {doi}")
        pmid = await _doi_to_pmid(doi)
        if pmid:
            status_lines.append(f"DOI → PMID: {pmid}")

    if ref_type == "pmcid":
        status_lines.append(f"Direct PMC access: {pmcid}")

    # --- Step 2: Try local PDF first if file path ---

    if ref_type == "file" and path:
        status_lines.append(f"Reading local PDF: {path}")
        text = _read_pdf_file(path)
        if text:
            return f"[Paper from local file: {path}]\n\n{text}\n\n---\nSource: local file"
        else:
            return (
                f"[Paper from local file: {path}]\n\n"
                f"Could not extract text from PDF. "
                f"Install pymupdf for PDF support: `pip install pymupdf`"
            )

    # --- Step 3: Try PubMed Central full text ---

    full_text = None

    if pmcid:
        status_lines.append(f"Fetching full text from PMC ({pmcid})")
        full_text = await _fetch_pmc_fulltext(pmcid)

    if not full_text and pmid:
        status_lines.append(f"Looking up PMCID for PMID {pmid}")
        pmcid = await _pmid_to_pmcid(pmid)
        if pmcid:
            status_lines.append(f"Found PMCID: {pmcid} — fetching full text")
            full_text = await _fetch_pmc_fulltext(pmcid)
        else:
            status_lines.append("No PMC full text available")

    if full_text:
        meta = await _pubmed_abstract(pmid) if pmid else None
        cite = ""
        if meta:
            cite = (
                f"\n---\n"
                f"**Source:** PMID:{pmid} | {pmcid} | "
                f"{meta['authors'][:60]} ({meta['year']}) {meta['journal']}"
            )
            if meta.get("doi"):
                cite += f"\n**DOI:** {meta['doi']}"
        return f"[Full text from PubMed Central — {pmcid}]\n\n{full_text}{cite}"

    # --- Step 4: Try Unpaywall for OA version ---

    if not doi and pmid:
        meta = await _pubmed_abstract(pmid)
        if meta and meta.get("doi"):
            doi = meta["doi"]

    if doi:
        status_lines.append(f"Checking Unpaywall for open access: {doi}")
        oa_url = await _unpaywall_lookup(doi)
        if oa_url:
            status_lines.append(f"Found OA version: {oa_url}")
            text = await _fetch_url_text(oa_url)
            if text:
                return (
                    f"[Open access version via Unpaywall]\n"
                    f"URL: {oa_url}\n\n{text}\n\n---\n"
                    f"Source: Unpaywall OA | DOI: {doi}"
                )
        else:
            status_lines.append("No open access version found")

    # --- Step 5: Try direct URL fetch ---

    if url and ref_type == "url":
        status_lines.append(f"Fetching URL: {url}")
        text = await _fetch_url_text(url)
        if text:
            return f"[Content from URL]\n{url}\n\n{text}"

    # --- Step 6: Fall back to abstract ---

    if pmid:
        status_lines.append("Falling back to abstract")
        meta = await _pubmed_abstract(pmid)
        if meta:
            lines = [
                "[Abstract only — full text not freely available]\n",
                f"# {meta['title']}\n",
                f"**Authors:** {meta['authors']}",
                f"**Journal:** {meta['journal']} ({meta['year']})",
                f"**PMID:** {meta['pmid']}",
            ]
            if meta["doi"]:
                lines.append(f"**DOI:** {meta['doi']}")
            lines.append(f"\n## Abstract\n\n{meta['abstract']}")

            lines.append(
                "\n---\n"
                "*Full text not available through open access channels. "
                "If you have a PDF, provide the file path and I can read it.*"
            )
            lines.append(f"\n*Resolution path: {' → '.join(status_lines)}*")

            return "\n".join(lines)

    # --- Nothing worked ---
    return (
        f"[Paper lookup: '{reference}']\n\n"
        f"Could not retrieve this paper. Tried: {' → '.join(status_lines)}\n\n"
        f"You can:\n"
        f"- Provide a PMID or DOI directly\n"
        f"- Share the PDF file path\n"
        f"- Try a more specific citation (e.g. 'Rapti 2019 Cell nerve ring')"
    )
