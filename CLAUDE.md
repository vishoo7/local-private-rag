# Personal RAG System

## Overview
A local-only RAG (Retrieval-Augmented Generation) system that indexes iMessage and Apple Mail data for semantic search and AI-powered querying. Everything runs on localhost — no data ever leaves the machine.

## Architecture

### Data Sources
- **iMessage**: SQLite database at `~/Library/Messages/chat.db` (~2.1GB, ~1.1M messages spanning 2008–present)
- **Apple Mail**: `.emlx` files at `~/Library/Mail/V10/` (~11GB including attachments, text content much smaller)

### AI Stack
- **Generation model**: Ollama `gemma3:4b` at `http://localhost:11434`
- **Embedding model**: Ollama `nomic-embed-text` (768-dimensional vectors)
- **Vector storage**: SQLite with `sqlite-vec` for ANN search (upgrade path from naive cosine similarity)

### Pipeline
1. **Ingest** — Extract text from iMessage DB and .emlx files, streaming in batches to avoid memory issues
2. **Chunk** — Group iMessages by contact + time window (4hr default). One chunk per email with metadata header.
3. **Embed** — Generate embeddings via Ollama and store in local SQLite vector DB
4. **Query** — Semantic search over embeddings, retrieve top-k chunks, generate answer with Gemma 3 4B
5. **Nightly cron** — Incremental update processing only new messages/emails from the last 24 hours

## Tech Stack
- Python 3.11+
- Ollama (must be running locally)
- SQLite / sqlite-vec
- numpy
- beautifulsoup4 (for HTML email parsing)
- No external APIs, no cloud services

## Privacy Rules for Claude Code Sessions
- **NEVER include actual message content, contact names, phone numbers, or email addresses in responses, commit messages, or any output that could be sent to Anthropic servers.**
- Do not paste, quote, or reference real user data in code comments, logs, or tool outputs.
- When debugging or testing, use synthetic/placeholder data only.
- This codebase handles private iMessage and email data — treat all ingested content as sensitive PII.

## Key Design Decisions
- **Privacy is paramount** — nothing leaves localhost, no telemetry, no external API calls
- **Stream/batch processing** — never load all 1.1M messages into memory at once; use generators and `fetchmany()`
- **Apple quirks** — iMessage timestamps are nanoseconds since 2001-01-01 (Core Data epoch). Convert: `unix_ts = date/1000000000 + 978307200`. `.emlx` files have a byte count on line 1 before standard RFC822 content.
- **Filter email noise** — skip noreply@, notifications@, marketing, newsletters before embedding to save space and improve retrieval quality
- **Chunking strategy** — iMessages grouped into conversation windows by contact + time proximity. Emails are one chunk each with subject/sender/date prefix for context.
- **Start small** — test with last 30 days of data before running full historical backfill

## Target Environment
- macOS on Apple Silicon (M-series)
- Terminal needs Full Disk Access (System Settings → Privacy & Security) to read chat.db and Mail directories
- Ollama running as a background service

## Project Structure
```
personal-rag/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── .env                    # paths, model names, config
├── src/
│   ├── __init__.py
│   ├── config.py           # load .env, constants
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── imessage.py     # streaming extraction from chat.db
│   │   └── email.py        # .emlx parsing from Apple Mail
│   ├── chunker.py          # conversation windowing, email chunking
│   ├── embed.py            # Ollama embedding + vector DB storage
│   ├── vectordb.py         # SQLite vector DB abstraction (search, insert, manage)
│   ├── query.py            # semantic search + Gemma generation
│   └── nightly.py          # incremental update script for cron
├── cli.py                  # CLI entry point: ingest, query, status commands
└── tests/
    ├── test_imessage.py
    ├── test_email.py
    └── test_query.py
```

## CLI Interface
```bash
# Initial setup — test with recent data first
python cli.py ingest --source imessage --since 30d
python cli.py ingest --source email --since 30d

# Full historical backfill (run overnight)
python cli.py ingest --source imessage
python cli.py ingest --source email

# Query
python cli.py query "What restaurant did Sarah recommend?"
python cli.py query "Open threads with John" --source imessage
python cli.py query "Find receipts from last month" --source email

# Status
python cli.py status  # show chunk counts, DB size, last update time

# Nightly incremental (called by cron)
python cli.py update
```

## Cron Setup
```bash
# crontab -e
0 3 * * * cd ~/projects/personal-rag && python cli.py update >> /tmp/personal-rag.log 2>&1
```

## Performance Expectations
- **Embedding speed**: ~50-100 chunks/min via Ollama on Apple Silicon
- **Full backfill**: Several hours for iMessage + email (run overnight)
- **Query latency**: <100ms retrieval + 5-20s generation = ~10-20s total
- **Disk usage**: ~2-5GB for extracted text + embeddings + indexes

## Web UI
- **Stack**: FastAPI + HTMX on `localhost:5391`
- **Query interface** with source filters and streaming responses
- **Ingestion controls** with progress tracking
- **Status dashboard** — chunk counts, DB size, last update, Ollama health
- **Background task management** for long-running ingestions

## Future Enhancements
- sqlite-vec or FAISS for faster ANN search at scale
- Daily digest generation — morning briefing of yesterday's important messages
- Contact resolution — map phone numbers/emails to names
- Attachment awareness — index filenames/metadata of attachments without storing them
- Conversation threading for email chains
