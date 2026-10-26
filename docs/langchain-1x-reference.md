# LangChain 1.x — Verified API Reference for This Project

> Generated 2026-09-05. **Every import below was verified by running it against this
> project's `.venv`**, not copied from a blog post. Re-verify with
> `docs/verify_imports.py` after any dependency bump.

## Installed versions (this venv)

| Package | Version |
|---|---|
| langchain | 1.4.0 |
| langchain-core | 1.6.2 |
| langchain-classic | 1.0.8 |
| langchain-community | 0.4.2 (**SUNSET — archived 2026-06-19**) |
| langchain-text-splitters | 1.1.2 |
| langchain-aws | 1.7.5 |
| langgraph | 1.2.11 |
| langchain-openai | 1.6.0 |
| ragas | 0.4.3 |
| faiss-cpu | 1.15.0 |
| rank-bm25 | 0.2.2 |
| datasets | 5.0.1 |
| tiktoken | 0.14.0 |

---

## 1. The three-package split (the #1 source of broken tutorials)

LangChain v1 shattered the old monolithic `langchain` package into three tiers.
Any tutorial written before Oct 2025 uses import paths that **no longer exist**.

| Tier | Package | What lives there |
|---|---|---|
| Primitives | `langchain-core` | `Document`, `BaseRetriever`, `Embeddings`, `BaseChatModel`, prompts, output parsers, LCEL/Runnable |
| Agent harness | `langchain` | `create_agent`, `init_chat_model`, `init_embeddings`, `@tool`, messages |
| Legacy attic | `langchain-classic` | Everything deprecated: old chains, **all retrievers**, indexing API, hub |
| Integrations | `langchain-<provider>` | One provider = one package (`langchain-aws`, `langchain-openai`, ...) |

### Verified: what exists and what does not

```
OK    from langchain.chat_models      import init_chat_model
OK    from langchain.embeddings       import init_embeddings
OK    from langchain.agents           import create_agent
OK    from langchain.tools            import tool
OK    from langchain_core.documents   import Document
OK    from langchain_core.vectorstores import InMemoryVectorStore
OK    from langchain_core.retrievers  import BaseRetriever
OK    from langchain_core.prompts     import ChatPromptTemplate
OK    from langchain_core.output_parsers import StrOutputParser
OK    from langchain_text_splitters   import RecursiveCharacterTextSplitter
OK    from langchain_text_splitters   import MarkdownHeaderTextSplitter
OK    from langchain_text_splitters   import TokenTextSplitter
OK    from langchain_aws              import ChatBedrockConverse, BedrockEmbeddings
OK    from langgraph.graph            import StateGraph

FAIL  import langchain.retrievers          -> ModuleNotFoundError (module deleted in v1)
OK    from langchain_classic.retrievers import EnsembleRetriever            # <- moved here
OK    from langchain_classic.retrievers import ContextualCompressionRetriever  # <- moved here

OK-but-SUNSET  from langchain_community.retrievers      import BM25Retriever
OK-but-SUNSET  from langchain_community.vectorstores    import FAISS
OK-but-SUNSET  from langchain_community.document_loaders import PyPDFLoader
```

**Consequence for this project:** the hybrid-search milestone needs
`EnsembleRetriever`, which every tutorial imports from `langchain.retrievers`.
That import raises `ModuleNotFoundError` here. Use `langchain_classic.retrievers`.

---

## 2. langchain-community is sunset

- Announced **2026-05-22**, repo archived **read-only 2026-06-19**.
- No phased deprecation; no drop-in replacements were published.
- Official guidance: migrate to dedicated per-provider packages, or
  **implement the component directly in application code**.
- Importing it emits a `DeprecationWarning` in this venv.

**Implication:** `BM25Retriever`, `FAISS`, and `PyPDFLoader` all currently come from
a dead package. For a portfolio/client project, three of our load-bearing components
sitting on an archived dependency is a real risk to name out loud.

Mitigation ranked by effort:
1. **BM25** — wrap `rank_bm25` in a ~25-line `BaseRetriever` subclass ourselves.
   Best option: it's also the clearest way to *learn* what BM25 actually does.
2. **Vector store** — `langchain_core.vectorstores.InMemoryVectorStore` for
   learning (zero deps, in `core`, not sunset); FAISS only when corpus size demands it.
3. **PDF loading** — call `pypdf` directly and build `Document` objects ourselves.
   ~15 lines, and it makes table-extraction behaviour visible instead of hidden.

---

## 3. RAG architectures (per official docs)

| Architecture | Control flow | Latency | When |
|---|---|---|---|
| **2-Step RAG** | Retrieve → generate, always, fixed | Predictable | **What we're building.** Baseline you measure against. |
| Agentic RAG | LLM decides when/whether/how to retrieve | Variable | Later, once the baseline is measured |
| Hybrid RAG | 2-step + query rewriting, retrieval validation, answer grading | Highest | After LangGraph |

Five building blocks the docs name: **Document Loaders → Text Splitters →
Embedding Models → Vector Stores → Retrievers**. Deliberately modular so any one
can be swapped without rewriting application logic — that modularity is exactly
what lets us A/B naive vs. hybrid vs. reranked.

### Note on the docs' own examples
The official retrieval pages lead with an **agentic** example (`create_agent` +
`@tool`), and give **no runnable 2-step RAG code** and **no stated defaults** for
chunk size or `k`. So there is no authoritative number to copy — every chunking
and `k` value in this project is a decision we make and justify, which is why
`notes/decisions.md` exists.

---

## 4. v1 agent API changes (for later, when we reach LangGraph)

- `langgraph.prebuilt.create_react_agent` → `langchain.agents.create_agent`
- `prompt=` → `system_prompt=`
- `pre_model_hook` / `post_model_hook` / `state_modifier` → middleware
  (`before_model`, `after_model`, `@wrap_model_call`, `@wrap_tool_call`)
- State schema must be `TypedDict` (Pydantic models / dataclasses rejected)
- Streaming node renamed `"agent"` → `"model"`
- `context=` replaces `config["configurable"]`
- `AIMessage.text` is now a **property**, not a method (`.text` not `.text()`)
- Chat models return `AIMessage`, narrowed from `BaseMessage`
- Python ≥ 3.10 required (we're on 3.12)

---

## 5. Model / embedding provider notes

`langchain-aws` is installed, implying Amazon Bedrock via `ChatBedrockConverse`.

**Blocker on record:** `aws sts get-caller-identity` fails — no credentials
configured, and `AWS_REGION` / `AWS_DEFAULT_REGION` are both unset.

**Important asymmetry to understand:** Anthropic publishes **no embedding model**.
A Claude-based RAG pipeline always uses a *different* vendor for the embedding
half. On Bedrock that means Amazon Titan Embeddings or Cohere Embed via
`BedrockEmbeddings`. Generation model and embedding model are independent choices
— this trips up a lot of people.

Current Anthropic model IDs (first-party API naming; Bedrock prefixes them):

| Model | ID | Context | In $/1M | Out $/1M |
|---|---|---|---|---|
| Claude Opus 5 | `claude-opus-5` | 1M | $5.00 | $25.00 |
| Claude Sonnet 5 | `claude-sonnet-5` | 1M | $2.00 | $10.00 |
| Claude Haiku 4.5 | `claude-haiku-4-5` | 200K | $1.00 | $5.00 |

Bedrock is partner-operated with **separate pricing** — see
https://aws.amazon.com/bedrock/pricing/ . Confirm the exact Bedrock model ID
string from the live Bedrock console rather than guessing; the prefixing and
cross-region-inference conventions change.

For RAGAS evaluation specifically, the judge model is a separate cost line from
the generation model, and evaluation replays every question — budget for it.

---

## Sources

- [LangChain overview](https://docs.langchain.com/oss/python/langchain/overview)
- [Retrieval / RAG architectures](https://docs.langchain.com/oss/python/langchain/retrieval)
- [Deep Agents retrieval](https://docs.langchain.com/oss/python/deepagents/retrieval)
- [v1 migration guide](https://docs.langchain.com/oss/python/migrate/langchain-v1)
- [langchain-community sunset notice](https://github.com/langchain-ai/langchain-community/issues/674)
- [EnsembleRetriever in langchain_classic](https://reference.langchain.com/python/langchain-classic/retrievers/ensemble/EnsembleRetriever)
- [Bedrock pricing](https://aws.amazon.com/bedrock/pricing/)
