# Cold Start Implementation Guide

How the cold start doctrine maps to code changes.

## New Files

| File | Purpose |
|------|---------|
| `gently/context/gap_assessment.py` | Gap assessor — inspects context store, identifies missing layers |
| `gently/capabilities/ingestion.py` | Ingestion capability — papers, PDFs, URLs, session history |
| `gently/daemon/onboarding.py` | Onboarding logic — generates questions, processes responses |

## Integration Points

### 1. Daemon startup (launch_copilot.py)

After creating the daemon and before starting it, run gap assessment and inject
onboarding tasks:

```python
# After daemon creation (line ~403)
from gently.context.gap_assessment import assess_gaps
from gently.daemon.onboarding import generate_onboarding_tasks

gap_report = assess_gaps(context_store)

if gap_report.conversation_weight != "none":
    onboarding_tasks = generate_onboarding_tasks(
        gap_report,
        session_id=copilot.session_id,
    )
    for task in onboarding_tasks:
        daemon.inject_task(task)

    console.print(
        f"  [{theme.muted}]Context: {gap_report.conversation_weight} "
        f"onboarding needed (readiness={gap_report.readiness:.0%})[/]"
    )
else:
    console.print(
        f"  [{theme.muted}]Context: ready "
        f"(readiness={gap_report.readiness:.0%})[/]"
    )
```

### 2. Ingestion capability wiring (launch_copilot.py)

Create ingestion alongside other capabilities:

```python
from gently.capabilities.ingestion import IngestionCapability

ingestion = IngestionCapability(
    claude_client=copilot.claude if not full_offline else None,
)
```

### 3. Ingestion capability in Capabilities class (capabilities.py)

Add ingestion as a fourth capability:

```python
class Capabilities:
    def __init__(self, ..., ingestion: Optional[IngestionCapability] = None):
        self.hardware = HardwareCapability(device_client)
        self.perception = PerceptionCapability(perception_manager)
        self.interaction = InteractionCapability(message_handler, notifier)
        self.ingestion = ingestion

    async def execute(self, action_type, params):
        match action_type:
            # ... existing cases ...
            case "ingest":
                source = params.get("source", "")
                if source.startswith(("http://", "https://")):
                    return await self.ingestion.ingest_url(source)
                elif source.endswith(".pdf"):
                    return await self.ingestion.ingest_pdf(source)
                else:
                    return await self.ingestion.ingest_text(source)
```

### 4. CLI command (rich_cli.py)

Register `/ingest` as a slash command:

```python
@command("ingest", "Ingest a paper, protocol, or URL into the daemon's context")
async def cmd_ingest(copilot, args):
    source = args.strip()
    if not source:
        console.print("Usage: /ingest <url, file path, or text>")
        return

    result = await copilot._daemon.capabilities.ingestion.ingest_url(source)
    # or ingest_pdf, ingest_text depending on source type

    console.print(f"Ingested: {result.summary}")
    console.print(f"  {result.entry_count} context entries extracted")

    # Apply extracted entries to context store
    apply_ingestion_result(result, copilot._daemon.context_store)
```

### 5. Onboarding response processing

When the daemon's ASK task gets a response, route it through onboarding
processing:

```python
# In scheduler.py, after interaction task execution
if task.task_type == TaskType.ASK and result.data.get("response"):
    from .onboarding import process_onboarding_response
    await process_onboarding_response(
        response=result.data["response"],
        topic=_infer_topic(task),
        context_store=self.context_store,
        claude_client=...,
        session_id=...,
    )
```

## Applying Ingestion Results to Context Store

Utility function that maps `IngestionResult` fields to context store operations:

```python
def apply_ingestion_result(result: IngestionResult, store: ContextStore):
    """Write extracted knowledge into the daemon's mind."""

    # Campaign proposal
    if result.campaign_proposal:
        store.create_campaign(
            description=result.campaign_proposal["description"],
            target=result.campaign_proposal.get("target"),
        )

    # Learnings
    for item in result.learnings:
        store.add_learning(Learning(
            id=gen_id(),
            content=item["content"],
            confidence=Confidence(item.get("confidence", "medium")),
            basis=f"ingestion:{result.source}",
        ))

    # Imaging parameters as learnings
    if result.imaging_parameters:
        params = result.imaging_parameters
        for key, value in params.items():
            if value is not None and key != "notes":
                store.add_learning(Learning(
                    id=gen_id(),
                    content=f"Recommended {key}: {value}",
                    confidence=Confidence.MEDIUM,
                    basis=f"ingestion:{result.source}",
                ))

    # Expectations
    for item in result.expectations:
        store.add_expectation(Expectation(
            id=gen_id(),
            target=item["target"],
            prediction=item["prediction"],
            expected_time=parse_timeframe(item.get("timeframe")),
            basis=f"ingestion:{result.source}",
        ))

    # Watchpoints
    for item in result.watchpoints:
        store.add_watchpoint(Watchpoint(
            id=gen_id(),
            target=item["target"],
            condition=item["condition"],
        ))
```

## Sequence at Cold Start

```
launch_copilot.py
    │
    ├── Create ContextStore (context.db)
    ├── Create Capabilities (incl. IngestionCapability)
    ├── Create Daemon
    │
    ├── assess_gaps(context_store)
    │   ├── Check learnings → Layer 1
    │   ├── Check campaigns → Layer 2
    │   ├── Check session_intents → Layer 3
    │   └── Return ContextGapReport
    │
    ├── generate_onboarding_tasks(gap_report)
    │   ├── Lab onboarding ASK (if first launch)
    │   ├── Campaign ASK (if no active campaign)
    │   └── Session intent ASK (if no intent)
    │
    ├── Inject tasks into daemon queue
    │
    ├── daemon.start()
    │   ├── Scheduler picks highest-priority onboarding task
    │   ├── Executes ASK → surfaces in CLI
    │   ├── Researcher responds
    │   ├── process_onboarding_response() → writes to context store
    │   ├── Next onboarding task (if any)
    │   └── Normal daemon loop begins with seeded context
    │
    └── run_rich_cli(copilot)
```

## What's NOT Included (Future Work)

- **Web search for related protocols**: Could use Claude's tool_use with a
  web search tool, or integrate with a search API.
- **Automatic session history ingestion**: Daemon could detect past sessions
  on first launch and offer to learn from them.
- **Campaign completion synthesis**: End-of-campaign summary generation.
- **Context export/import**: Moving context between installations.
- **Multi-researcher support**: Different context profiles for different users.
