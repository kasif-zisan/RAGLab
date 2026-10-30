# Privacy — what leaves the network, and why vectors are not anonymous

## 1. The fact people get wrong

**An embedding is not anonymised data. Vectors can leak their source text.**

There is a body of research on *embedding inversion*: given the vectors and
access to the same model, a meaningful amount of the original text can be
reconstructed. Not always perfectly, but well enough that treating a vector
store as "just numbers" is wrong.

Practical consequence, and it is the whole point of this note:

> **A vector store derived from sensitive documents is itself sensitive.** It
> needs the same access control, encryption, retention policy, audit logging and
> deletion handling as the source documents.

Where this goes wrong in practice: the source PDFs sit in a locked-down bucket
with careful IAM, and the vectors end up in a convenient managed service with a
shared API key, because "they're only numbers". That is a data leak with extra
steps.

The same applies to anything derived: the chunk text archive, the cached query
vectors, the logs.

---

## 2. Three tiers of where the text goes

| tier | examples | what you tell the client |
|---|---|---|
| your own hardware | EC2, on-prem GPU | never leaves our network |
| API inside their cloud | Bedrock, Vertex, Azure OpenAI | stays in our account, in our region |
| external API | OpenAI, Voyage, Cohere direct | leaves our network; needs a DPA |

For finance, legal, health and government work this question is asked **before**
quality, and it eliminates options regardless of benchmark scores. It is the
main practical reason to know Bedrock: same class of model, inside the client's
own account, which usually ends the procurement conversation.

---

## 3. What to ask the client, in writing

1. **Does this corpus contain personal data?** Names, IDs, health, financial,
   employment records. In long documents the answer is usually yes somewhere.
2. **Which jurisdiction, and may data leave it?** Data residency is often a hard
   boundary that picks your region and rules out providers.
3. **Is there a deletion obligation?** GDPR-style erasure covers derived data.
   See `index-updates.md` — this is why delete has to work, and why "tombstoned
   but still on disk" may not satisfy it.
4. **What is the retention period?** Including backups and old index versions.
5. **Who may query it?** If some users must not see some documents, you need
   per-user filtering at retrieval time — and that is a real architectural
   requirement, not a UI detail. It is also one of the strongest reasons to use
   a vector database, which can filter on metadata.

Get the answers written down. They change the architecture, and "you never told
me it was medical data" is not a conversation you want to have later.

---

## 4. Things that quietly leak

- **Logs.** Chunk text in a debug log, or user questions in an access log, both
  become copies of the sensitive data outside your controls. Decide what gets
  logged before you ship, not after an incident.
- **Model-provider retention.** Some providers retain inputs for abuse
  monitoring by default. Check the terms, and check whether a zero-retention or
  enterprise tier exists.
- **Sub-processors.** A managed vector database runs on someone else's cloud.
  That is a sub-processor the client may be obliged to disclose.
- **Your own machine.** The 280 MB of vectors you pulled down to test with, and
  the chunk archive you unzipped into a temp directory, are the client's data
  sitting on a laptop. Have a deletion step at the end of the engagement and say
  so in the contract.
- **Per-user access.** If retrieval ignores permissions, your RAG system becomes
  a way to read documents a user could not otherwise open. This is the most
  common serious design flaw in corporate RAG deployments.

---

## 5. A minimum checklist for a sensitive engagement

- Source documents, chunk archive and vectors are all in storage with the same
  access controls, encrypted at rest.
- Embedding happens inside the client's cloud or on their hardware, unless they
  have signed off on an external provider.
- A written DPA with every provider that sees the text, sub-processors listed.
- Deletion is implemented and **tested** end to end: source, chunks, vectors,
  caches, backups.
- Retrieval filters on the querying user's permissions.
- Logging policy decided in writing, with text and queries excluded by default.
- Your local copies are deleted at the end of the engagement, and you say so.

None of this is hard. It is just invisible until someone asks, and by then the
architecture is already built.
