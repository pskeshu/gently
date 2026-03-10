# Slash Command Reference

All commands are registered in `gently/harness/commands.py`.

## Navigation

| Command | Description |
|---------|-------------|
| `/quit` | Exit the agent (aliases: `/exit`, `/q`) |
| `/clear` | Clear screen and show welcome banner |
| `/help [command]` | Show help for all commands or a specific command |

## Inspection

| Command | Description |
|---------|-------------|
| `/status` | Show experiment status (microscope connection, active embryos, detectors) |
| `/detectors` | List all registered detectors with status, type, and configuration |
| `/embryos [ID]` | List embryos or show details for a specific embryo |
| `/timelapse [watch]` | Timelapse acquisition status; `watch` for live countdown view |
| `/timeline [clear] [--filter TYPE] [--embryo ID] [--since TIME] [--all]` | Event timeline with filtering and multiple view modes (`--letters`, `--log`, `--table`, `--axis`) |
| `/peers [HOSTNAME] [campaigns]` | Show mesh peers on the network (alias: `/mesh`) |

## Session

| Command | Description |
|---------|-------------|
| `/sessions` | Browse saved sessions interactively |
| `/resume [ID]` | Resume a previously saved session (interactive picker if no ID) |
| `/save` | Save current session with embryo states and conversation history |
| `/import-embryos [ID\|last]` | Import embryo definitions from another session |
| `/make-video [embryo_id] [--fps N]` | Create MP4 video from timelapse volumes |
| `/wizard` | Re-run the onboarding wizard |

## Planning

| Command | Description |
|---------|-------------|
| `/plan [status\|exit]` | Enter plan mode for experimental design, or show plan status |
| `/campaign [delete\|share\|unshare] [ID]` | View, manage, and share campaigns (alias: `/campaigns`) |
| `/join-campaign <HOSTNAME> <CAMPAIGN_ID>` | Join a shared campaign on a peer |
| `/claim <ITEM_ID>` | Claim a plan item from a joined remote campaign |
| `/pair [accept\|reject\|list\|unpair\|scopes] [target]` | Bluetooth-style pairing with mesh peers |

## Appearance

| Command | Description |
|---------|-------------|
| `/theme [name]` | Switch color theme (vibrant, scientific, claude, monochrome) |
| `/history` | Show conversation history with the agent |
| `/tokens` | Show API token usage statistics and estimated cost |

## Diagnostics

| Command | Description |
|---------|-------------|
| `/benchmark [--volumes N] [--slices N] [--warmup N] [--save]` | Run end-to-end volume acquisition FPS benchmark |
| `/reset-context` | Clear the context database (campaigns, learnings, session intents) |
