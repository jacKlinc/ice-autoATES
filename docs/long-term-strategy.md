**Phase 1 — Ship the foundation (Now → June)**

Get the AutoATES-v2.0 integration working end-to-end. GPX input → DEM processing → ATES classification → output. Clean README, GPL-3.0 licence, working code. This is the core that everything else builds on.

**Phase 2 — Add ML credibility (June → August)**

Wrap the AutoATES classifier in MLflow experiment tracking. Log parameters, metrics, and model versions. This is a relatively small addition but transforms the project from "runs a script" to "has a real ML pipeline." Adds a genuine AI engineering signal to your CV while you're employed and the PR clock is running.

**Phase 3 — Vector database layer (August → November)**

Ingest the ice climbing atlas — route descriptions, grades, approach notes, hazard information — into a vector database (pgvector is the pragmatic choice, keeps it in Postgres which you know). Build a retrieval layer that can answer queries like "WI4 routes near Canmore with low avalanche exposure." This is the RAG foundation.

**Phase 4 — LLM frontend (Winter onwards)**

Build a natural language interface over the retrieval layer. A user describes a route objective, the system returns terrain classification plus relevant atlas entries. This is the full vision — and by this point you'll have the skills, the data, and the infrastructure to build it properly.
