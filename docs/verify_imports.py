"""Re-verify the import paths recorded in docs/langchain-1x-reference.md.

Run after any dependency bump:  uv run python docs/verify_imports.py
Exits non-zero if an import we depend on has moved again.
"""

import importlib
import importlib.metadata as md
import sys

EXPECTED_OK = [
    ("langchain.chat_models", "init_chat_model"),
    ("langchain.embeddings", "init_embeddings"),
    ("langchain.agents", "create_agent"),
    ("langchain.tools", "tool"),
    ("langchain_core.documents", "Document"),
    ("langchain_core.vectorstores", "InMemoryVectorStore"),
    ("langchain_core.retrievers", "BaseRetriever"),
    ("langchain_core.prompts", "ChatPromptTemplate"),
    ("langchain_core.output_parsers", "StrOutputParser"),
    ("langchain_text_splitters", "RecursiveCharacterTextSplitter"),
    ("langchain_text_splitters", "MarkdownHeaderTextSplitter"),
    ("langchain_text_splitters", "TokenTextSplitter"),
    ("langchain_classic.retrievers", "EnsembleRetriever"),
    ("langchain_classic.retrievers", "ContextualCompressionRetriever"),
    ("langchain_aws", "ChatBedrockConverse"),
    ("langchain_aws", "BedrockEmbeddings"),
    ("langgraph.graph", "StateGraph"),
]

# Documented as gone in v1. If one of these starts working, the docs are stale.
EXPECTED_GONE = [
    ("langchain.retrievers", "EnsembleRetriever"),
]

PACKAGES = [
    "langchain", "langchain-core", "langchain-classic", "langchain-community",
    "langchain-text-splitters", "langchain-aws", "langgraph", "ragas",
    "faiss-cpu", "rank-bm25", "datasets", "tiktoken",
]


def check(mod_name: str, attr: str) -> bool:
    try:
        mod = importlib.import_module(mod_name)
    except ImportError:
        return False
    return hasattr(mod, attr)


def main() -> int:
    print("== installed versions ==")
    for pkg in PACKAGES:
        try:
            print(f"  {pkg:28} {md.version(pkg)}")
        except md.PackageNotFoundError:
            print(f"  {pkg:28} (not installed)")

    failures = []

    print("\n== imports expected to work ==")
    for mod_name, attr in EXPECTED_OK:
        ok = check(mod_name, attr)
        print(f"  {'ok  ' if ok else 'FAIL'}  from {mod_name} import {attr}")
        if not ok:
            failures.append(f"{mod_name}.{attr} no longer importable")

    print("\n== imports expected to be gone ==")
    for mod_name, attr in EXPECTED_GONE:
        gone = not check(mod_name, attr)
        print(f"  {'ok  ' if gone else 'DRIFT'}  {mod_name}.{attr} absent")
        if not gone:
            failures.append(f"{mod_name}.{attr} came back — reference doc is stale")

    if failures:
        print("\nDRIFT DETECTED — update docs/langchain-1x-reference.md:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nAll import paths match docs/langchain-1x-reference.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
