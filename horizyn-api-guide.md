# Horizyn API Guide

The Horizyn API is a hosted service that searches a large library of natural
enzymes for the ones most likely to catalyze a reaction of interest. You supply a
reaction as a SMILES string; the API returns a ranked list of candidate enzymes
with annotations (name, organism, EC numbers, cofactors, sequence length, and
more).

Anyone can get a key — it takes one email round-trip and no account or password.

**You do not need to clone this repository to use the API.** It is plain HTTP and
JSON against a hosted service, so a key and `curl` — or any HTTP client in any
language — is the whole toolchain. Nothing below installs anything from here, and
none of it needs a GPU or a local data download. The repository is for training
the model or running it offline yourself; the two are independent.

**Base URL:**

```
https://api.horizyn1.dayhofflabs.com
```

**Interactive API docs** (OpenAPI / Swagger UI, no key required to browse):

```
https://api.horizyn1.dayhofflabs.com/docs
```

The machine-readable schema is at `/openapi.json`, which is handy for generating
a client in your language of choice.

## Table of Contents

- [Horizyn API Guide](#horizyn-api-guide)
  - [Table of Contents](#table-of-contents)
  - [1. Hosted API vs. this repository](#1-hosted-api-vs-this-repository)
  - [2. Get an API key](#2-get-an-api-key)
    - [Step 1 — request a verification code](#step-1--request-a-verification-code)
    - [Step 2 — confirm and receive your key](#step-2--confirm-and-receive-your-key)
    - [Things worth knowing](#things-worth-knowing)
  - [3. Authenticate](#3-authenticate)
  - [4. Search for enzymes by reaction](#4-search-for-enzymes-by-reaction)
    - [Request fields](#request-fields)
    - [Response (flat list)](#response-flat-list)
    - [`warnings`: a filter term that can match nothing names itself](#warnings-a-filter-term-that-can-match-nothing-names-itself)
    - [Response (clustered)](#response-clustered)
  - [5. List available screening sets](#5-list-available-screening-sets)
  - [5a. List available cofactors](#5a-list-available-cofactors)
  - [6. Validate SMILES before querying](#6-validate-smiles-before-querying)
  - [7. Look up an enzyme](#7-look-up-an-enzyme)
  - [8. Rate limits](#8-rate-limits)
    - [One allowance per key, shared across endpoints](#one-allowance-per-key-shared-across-endpoints)
    - [Request body size](#request-body-size)
    - [Capacity](#capacity)
  - [9. Errors](#9-errors)
  - [10. Python example](#10-python-example)
    - [One-time: get a key](#one-time-get-a-key)
    - [Querying](#querying)
    - [Inspecting a hit](#inspecting-a-hit)
  - [Questions](#questions)

---

## 1. Hosted API vs. this repository

The two are complementary, and neither needs the other. **Using the API requires
nothing from this repository** — no clone, no install, no download.

| | This repository | Hosted API |
|---|---|---|
| **Enzymes searched** | ~216K bundled protein embeddings | 6.33M proteins |
| **Setup** | Install deps, download ~1GB data + checkpoint | Request a key by email; nothing to install |
| **Hardware** | NVIDIA GPU with 16GB+ VRAM | None — runs server-side |
| **Annotations** | Not included | Name, organism, EC, cofactors, literature, structures |
| **Filtering / clustering** | Not included | EC, transporter class, cofactor, length; EC clustering |
| **Retraining, custom data** | Yes | No |
| **Offline / no rate limit** | Yes | 60 req/min, 10,000 req/month per key |

Use the repository to reproduce the paper, retrain the model, or run fully
offline. Use the API to screen against far more enzymes than the bundled set,
with metadata and filtering, and without provisioning a GPU.

---

## 2. Get an API key

Keys are issued through an email-verification flow. No account or password is
required — just an email address you control.

### Step 1 — request a verification code

```bash
curl -X POST https://api.horizyn1.dayhofflabs.com/keys/request \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com"}'
```

The response is always the same, whether or not the address is deliverable:

```json
{"message": "If this email is valid, a verification email has been sent."}
```

Check your inbox for a short verification code, sent from
`noreply@dayhofflabs.com`.

### Step 2 — confirm and receive your key

```bash
curl -X POST https://api.horizyn1.dayhofflabs.com/keys/confirm \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com", "code": "123456", "name": "my-project"}'
```

```json
{"api_key": "hzk_live_xxxxxxxx..."}
```

Store this key securely — it is shown **only once** and cannot be retrieved
later. If you lose it, request a new one. The optional `name` field just labels
the key for your own reference.

### Things worth knowing

- The verification code **expires 15 minutes** after it is sent.
- You must wait **60 seconds** between verification requests for the same email
  address, and one source address may request at most **2 codes per hour**. A
  request inside either window still returns the `202` body above, but no email
  is sent — so if nothing arrives, wait rather than retrying immediately.
- After **5 incorrect** code attempts the code is invalidated — request a new
  one. `/keys/confirm` also accepts at most **10 attempts per email per hour**;
  beyond that it returns `429` with a `Retry-After` header.
- Treat the key like a password. Keep it out of source control; prefer an
  environment variable, as in the examples below.

---

## 3. Authenticate

Send your key as a bearer token on every data request:

```
Authorization: Bearer hzk_live_xxxxxxxx...
```

The two `/keys/*` endpoints above need no key, and neither do `/healthz`, `/docs`
and `/openapi.json`. Every other request without a valid key returns
`401 Unauthorized`.

Repeated failed authentications are budgeted per source address: after 120
failures in a rolling minute, that address receives `429` with `Retry-After`
instead of `401` until the window rolls over.

---

## 4. Search for enzymes by reaction

`POST /query/reaction` is the primary endpoint. Give it a reaction SMILES and it
returns ranked enzyme candidates.

**Minimal request:**

The reaction below is GabT transamination of 2-oxoglutarate using
6-aminohexanoate — a real, atom- and charge-balanced reaction. Balance your
reactions: include cosubstrates and cofactor-derived products rather than writing
only the transformation you care about. Unbalanced input still returns results,
but the closer your SMILES is to the full biochemical reaction, the better the
matches.

```bash
curl -X POST https://api.horizyn1.dayhofflabs.com/query/reaction \
  -H "Authorization: Bearer $HORIZYN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "[NH3+]CCCCCC([O-])=O.[O-]C(=O)CCC(=O)C([O-])=O>>[O-]C(=O)CCCCC=O.[NH3+][C@@H](CCC([O-])=O)C([O-])=O"}'
```

### Request fields

Every field carries an explicit limit, and exceeding one returns `422` naming the
field. Unknown fields are rejected rather than ignored.

| Field | Type | Default | Limit | Description |
|-------|------|---------|-------|-------------|
| `smiles` | string | — (required) | 2048 chars total, 1024 per side, 768 per component | Reaction SMILES, using `>>` between reactants and products |
| `screening_set` | string | default set | 46 chars | Which enzyme library to search. Omit it or send `null` to use the default; see [§ 5](#5-list-available-screening-sets) |
| `top_k` | int | `100` | **1–1000** | How many nearest enzymes to retrieve before filtering/paging |
| `page` | int | `1` | 1–1000 | Page number of results to return |
| `page_size` | int | `20` | 1–1000 | Results per page |
| `cluster_by` | string | `null` | `"ec3"` or `"ec4"` | Group results by EC number instead of returning a flat list: `"ec3"` (first three EC fields) or `"ec4"` (all four). **Any other value returns `422`** |
| `cluster_size` | int | `5` | 1–1000 | Members shown per cluster when clustering |
| `fetch_metadata` | bool | `true` | — | Attach enzyme annotations (name, organism, EC, etc.) to results |
| `ec_include` / `ec_exclude` | list[string] | `null` | ≤190 terms, ≤110 chars each | Keep / drop results by EC number prefix |
| `tc_include` / `tc_exclude` | list[string] | `null` | ≤190 terms, ≤110 chars each | Keep / drop results by transporter classification |
| `cofactor_include` / `cofactor_exclude` | list[string] | `null` | ≤190 terms, ≤110 chars each | Keep / drop results by cofactor. Accepts acronyms (`"PLP"`, `"SAM"`, `"TPP"`), element names (`"zinc"`), and ChEBI spellings; case-insensitive |
| `min_length` / `max_length` | int | `null` | 0–90,708 | Keep results within a sequence-length range. `min_length > max_length` returns `422` rather than an empty result set |

`top_k` is applied *before* filtering, so aggressive filters combined with a small
`top_k` can return very few rows — raise it, up to the ceiling of 1000.

**That ceiling of 1000 is a hard one, and paging is not a way around it.** `page`
and `page_size` slice *within* the `top_k` pool rather than extending it, so 1000
rows is the largest result set any single query can produce. A larger `top_k`
returns `422`; if you previously used `top_k: 5000`, that is a breaking change
with no client-side substitute.

Cofactor filters accept common acronyms and names as well as the ChEBI spellings
used in the annotations, and matching is case-insensitive. All of `"PLP"`,
`"pyridoxal phosphate"` and `"pyridoxal 5'-phosphate"` select the same
PLP-dependent enzymes; `"SAM"`, `"TPP"`, `"PQQ"`, `"B12"`, `"zinc"` and `"Heme"`
work too. Call `GET /cofactors` to list every annotated cofactor with enzyme
counts. EC and transporter-class filters match by prefix instead.

Two caveats specific to cofactor filters: annotations are sparse — roughly two
thirds of enzymes have no cofactor annotation at all, and a `cofactor_include`
filter drops every one of them — and filtering happens after the `top_k` search,
so pair a narrow cofactor filter with a large `top_k`.

### Response (flat list)

```json
{
  "results": [
    {
      "id": "P22256",
      "score": 0.915,
      "name": "4-aminobutyrate aminotransferase GabT",
      "organism": "Escherichia coli (strain K12)",
      "ec_numbers": ["2.6.1.19", "2.6.1.48"],
      "cofactors": ["pyridoxal 5'-phosphate"],
      "expression_score": 0.7273,
      "tc_numbers": [],
      "organism_lineage": ["Bacteria", "Pseudomonadati", "Pseudomonadota"],
      "length": 426
    }
  ],
  "filter_stats": { "...": "counts before/after each filter" },
  "timings_ms": { "total": 142.0 },
  "page": 1,
  "page_size": 20,
  "total_results": 100,
  "total_pages": 5,
  "quality_warnings": [],
  "warnings": [],
  "metadata_fetched": true
}
```

`score` is a cosine similarity in `[0, 1]`; higher means the enzyme is a closer
match for the reaction. Scores are best read as a *ranking* — compare them
against each other within a single query rather than treating them as absolute
measures. A high score suggests biochemical relevance; it is a hypothesis to
test, not evidence of catalytic activity.

`id` is the UniProt accession, which you can pass to the protein endpoints in
section 7.

### `warnings`: a filter term that can match nothing names itself

An empty result set is ambiguous — it can mean "nothing scored well enough" or
"you filtered on a term this library has never heard of". A filter term that
matches **no value anywhere in the screening set** is named in `warnings` instead
of silently returning zero rows:

```json
{
  "warnings": [
    "ec_include: no protein in this screening set is annotated with EC '9.9.9'. EC numbers are hierarchical, so a prefix such as '1.1' matches every number beneath it.",
    "cofactor_include: no cofactor in this screening set matches 'unobtainium'. Cofactors are annotated with ChEBI names; see GET /cofactors for the values available."
  ]
}
```

This covers `ec_include`/`ec_exclude`, `tc_include`/`tc_exclude`, and
`cofactor_include`/`cofactor_exclude`. The check runs against the set's distinct
annotated values, so a warning means "this term can never match in this library" —
not "it happened to match nothing among this query's top hits", which is a normal
outcome and is not warned about. `filter_stats` reports how many results each
filter stage kept.

### Response (clustered)

If you set `cluster_by`, results are grouped by EC number instead. This helps
when the top hits are dominated by many near-identical homologs and you want a
per-function overview:

```json
{
  "clusters": [
    {
      "ec_labels": ["2.6.1.19"],
      "cluster_key": "2.6.1.19",
      "best_score": 0.915,
      "total_in_cluster": 12,
      "members": [ { "id": "P22256", "score": 0.915 } ]
    }
  ],
  "total_clusters": 34,
  "page": 1,
  "page_size": 20,
  "total_pages": 2,
  "filter_stats": { "...": "counts before/after each filter" },
  "timings_ms": { "total": 150.0 },
  "quality_warnings": [],
  "warnings": [],
  "metadata_fetched": true
}
```

Results with no usable EC annotation are grouped under an empty cluster key.
Pagination applies to clusters rather than individual enzymes. `cluster_by`
accepts only `"ec3"` and `"ec4"`; anything else returns `422` rather than
falling back to `"ec3"` and quietly answering a different question.

---

## 5. List available screening sets

A *screening set* is the enzyme library your reaction is compared against. There
is currently one, and it is the default — so you can omit `screening_set`
entirely and every example in this guide will work.

To see what is available, and to discover the id if you want to name one
explicitly:

```bash
curl https://api.horizyn1.dayhofflabs.com/screening_sets \
  -H "Authorization: Bearer $HORIZYN_API_KEY"
```

```json
{
  "screening_sets": [
    {
      "screening_set_id": "...",
      "screening_set_type": "protein",
      "description": "Natural enzymes from UniProt with ProtT5 embeddings",
      "rows": 6332375,
      "dim": 512,
      "is_default": true
    }
  ]
}
```

The `screening_set_id` and `description` returned by the live API are
authoritative; read them from the response rather than hard-coding either, since
the available sets may change. Pass a `screening_set_id` as the `screening_set`
field on a query, or as the `screening_set` query parameter on the protein
endpoints, to target a specific set. The one flagged `is_default` is used when
you omit it.

---

## 5a. List available cofactors

`GET /cofactors` returns every cofactor annotated in a screening set, with the
number of enzymes carrying it, most common first. Use it to discover exactly what
`cofactor_include` / `cofactor_exclude` can filter on.

```bash
curl https://api.horizyn1.dayhofflabs.com/cofactors \
  -H "Authorization: Bearer $HORIZYN_API_KEY"
```

```json
{
  "screening_set": "...",
  "cofactors": [
    {"name": "Mg(2+)", "count": 857150},
    {"name": "Zn(2+)", "count": 342243},
    {"name": "pyridoxal 5'-phosphate", "count": 208973},
    {"name": "FAD", "count": 133685}
  ]
}
```

Annotations come from UniProt and use ChEBI names, so the stored value is
`pyridoxal 5'-phosphate` rather than `PLP`. Filters accept either, along with
other common acronyms (`"SAM"`, `"TPP"`, `"PQQ"`, `"B12"`) and element names
(`"zinc"`), case-insensitively — but this endpoint is the authoritative list of
what is actually annotated.

Filtering applies to the `top_k` nearest enzymes rather than the whole library, so
pair a narrow cofactor filter with a large `top_k`. The maximum is **1000**, which
is therefore also the most any filter has to work with.

---

## 6. Validate SMILES before querying

These endpoints check that your input parses, without running a search. They are
useful for giving users fast feedback in an interactive tool, and cheaper than
discovering a typo via a `400` on a full query.

Both take the same `smiles` length limits as a query — 2048 chars total, 1024 per
side, 768 per component — and both draw on the same per-key allowance as a search
(see [§ 8](#8-rate-limits)).

**Reaction:**

```bash
curl -X POST https://api.horizyn1.dayhofflabs.com/validate/reaction \
  -H "Authorization: Bearer $HORIZYN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "[NH3+]CCCCCC([O-])=O.[O-]C(=O)CCC(=O)C([O-])=O>>[O-]C(=O)CCCCC=O.[NH3+][C@@H](CCC([O-])=O)C([O-])=O"}'
```

```json
{"valid": true, "parseable": true, "quality_warnings": [], "atom_balance": null}
```

**Compound:**

```bash
curl -X POST https://api.horizyn1.dayhofflabs.com/validate/compound \
  -H "Authorization: Bearer $HORIZYN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CCO"}'
```

```json
{"valid": true, "smiles": "CCO", "canonical_smiles": "CCO", "error": null}
```

---

## 7. Look up an enzyme

Once a query gives you an enzyme `id` (UniProt accession), you can fetch more
about it. Both endpoints search the default screening set unless you pass a
`screening_set` query parameter.

**Annotations from the screening set:**

```bash
curl "https://api.horizyn1.dayhofflabs.com/protein/P22256/details" \
  -H "Authorization: Bearer $HORIZYN_API_KEY"
```

```json
{
  "protein_id": "P22256",
  "name": "4-aminobutyrate aminotransferase GabT",
  "ec_numbers": ["2.6.1.19", "2.6.1.48"],
  "organism": "Escherichia coli (strain K12)",
  "cofactors": ["pyridoxal 5'-phosphate"],
  "expression_score": 0.7273,
  "tc_numbers": [],
  "organism_lineage": ["Bacteria", "Pseudomonadati", "Pseudomonadota"],
  "length": 426,
  "metadata_source": "parquet"
}
```

**Literature and structure references** (fetched live from UniProt):

```bash
curl "https://api.horizyn1.dayhofflabs.com/protein/P22256/references" \
  -H "Authorization: Bearer $HORIZYN_API_KEY"
```

```json
{
  "description": "4-aminobutyrate aminotransferase GabT",
  "annotation_score": 5.0,
  "literature": [
    {"title": "...", "citation_type": "journal article", "journal": "...", "publication_date": "1998", "pubmed_id": "..."}
  ],
  "pdb_ids": ["1ABC"],
  "fetched_at": "2026-07-01T00:00:00+00:00"
}
```

Responses may gain additional fields over time; parse defensively and ignore
keys you do not recognize.

The `{id}` path segment is constrained to `[A-Za-z0-9._:-]`, at most 82
characters; anything else returns `422` without a lookup.

---

## 8. Rate limits

### One allowance per key, shared across endpoints

Each key is limited to:

- **60 requests per minute**
- **10,000 requests per month**

This is **one shared allowance across all seven authenticated endpoints**, not a
separate budget per endpoint. `/query/reaction`, `/validate/reaction`,
`/validate/compound`, `/screening_sets`, `/cofactors`, `/protein/{id}/details`
and `/protein/{id}/references` all draw on the same counter, so a cheap
`/screening_sets` call spends the same quota as a search.

Exceeding either limit returns `429 Too Many Requests`, with a `Retry-After`
header in seconds and a body like:

```json
{"error": "rate limited", "retry_after": 42}
```

Quota is charged **before** the request body is validated on every enveloped
endpoint except `/query/reaction`, so a request that fails schema validation with
a `422` has already spent a unit. Fix client-side validation errors rather than
retrying them in a loop.

### Request body size

Request bodies are capped at **512 KiB** (524,288 bytes). A larger body returns
`413 Payload Too Large`, enforced on bytes actually received rather than on the
`Content-Length` you declare. A client that announces a body and then stops
sending is disconnected with `408 Request Timeout` after 30 seconds.

### Capacity

The service runs a single GPU worker, so it also sheds load when saturated,
independently of your own allowance:

| Response | Cause |
|----------|-------|
| `429` | The concurrency gate or the fingerprinting pool is full. Retry after `Retry-After`. |
| `503` | A fingerprinting worker died, or the rate-limit counter store could not be reached. Both are transient and retryable — the service fails closed rather than admitting a request it cannot account for. |
| `504` | The query exceeded its 60-second budget. Retry, optionally with a smaller `top_k`. |

In every case, wait `Retry-After` seconds (where present) and retry.

If your work needs more than these limits, get in touch at
[info@dayhofflabs.com](mailto:info@dayhofflabs.com).

---

## 9. Errors

| Status | Meaning |
|--------|---------|
| `400` | Invalid input (e.g. malformed reaction SMILES). Body includes `quality_warnings`. Also returned for an incorrect verification code. |
| `401` | Missing or invalid API key. |
| `404` | Unknown screening set, or protein not found. |
| `408` | A declared request body stopped arriving mid-send. |
| `413` | Request body exceeded 512 KiB. |
| `422` | Request failed validation — a missing required field, a wrong type, an unknown field, a value past one of the limits in [§ 4](#4-search-for-enzymes-by-reaction), an unrecognized `cluster_by`, or an unusable email address on `/keys/*`. The body names the offending field. |
| `429` | Rate limited, server briefly at capacity, or too many failed authentications — see `Retry-After`. |
| `503` | Service not ready, a worker died, the rate-limit counter store is unreachable, or verification email is temporarily unavailable. Retry shortly. |
| `504` | Query timed out. Retry, optionally with a smaller `top_k`. |

Error bodies name the field and the limit but never echo the value you sent, so a
`422` on a large payload stays small.

Malformed, oversized, and pathological input is answered with a `4xx`. If you do
receive a `500`, it is a bug — please report it with the request that caused it.

---

## 10. Python example

The API is plain HTTP + JSON, so no SDK is needed and nothing from this
repository is involved. These examples need only
[`requests`](https://requests.readthedocs.io/) in whatever environment you
already have:

```bash
pip install requests
```

### One-time: get a key

```python
import requests

BASE = "https://api.horizyn1.dayhofflabs.com"
EMAIL = "you@example.com"

requests.post(f"{BASE}/keys/request", json={"email": EMAIL}, timeout=30).raise_for_status()
code = input("Verification code from your email: ").strip()

resp = requests.post(
    f"{BASE}/keys/confirm",
    json={"email": EMAIL, "code": code, "name": "my-project"},
    timeout=30,
)
resp.raise_for_status()
print("Save this key somewhere safe:", resp.json()["api_key"])
```

### Querying

```python
import os
import time

import requests

BASE = "https://api.horizyn1.dayhofflabs.com"
API_KEY = os.environ["HORIZYN_API_KEY"]

session = requests.Session()
session.headers.update({"Authorization": f"Bearer {API_KEY}"})


def query_reaction(smiles: str, **options) -> dict:
    """Search for enzymes that may catalyze `smiles`, retrying on rate limits."""
    payload = {"smiles": smiles, **options}

    for _ in range(5):
        resp = session.post(f"{BASE}/query/reaction", json=payload, timeout=120)
        if resp.status_code in (429, 503):
            time.sleep(int(resp.headers.get("Retry-After", 5)))
            continue
        resp.raise_for_status()
        return resp.json()

    raise RuntimeError("still being refused after 5 attempts")


# PLP-dependent transaminases for the GabT reaction, small enough to be
# practical to express. `top_k` is at its 1000 maximum because these filters are
# narrow — see the note in section 4.
data = query_reaction(
    "[NH3+]CCCCCC([O-])=O.[O-]C(=O)CCC(=O)C([O-])=O"
    ">>[O-]C(=O)CCCCC=O.[NH3+][C@@H](CCC([O-])=O)C([O-])=O",
    top_k=1000,
    page_size=10,
    cofactor_include=["PLP"],
    max_length=600,
)

# Anything here means a filter term matched nothing in the library — check it
# before concluding that no enzyme fits.
for warning in data["warnings"]:
    print("warning:", warning)

for hit in data["results"]:
    print(f"{hit['score']:.3f}  {hit['id']:10s}  {hit['name']} ({hit['organism']})")
```

### Inspecting a hit

```python
top_id = data["results"][0]["id"]

details = session.get(f"{BASE}/protein/{top_id}/details", timeout=30).json()
print(details)

refs = session.get(f"{BASE}/protein/{top_id}/references", timeout=60).json()
for paper in refs.get("literature", [])[:5]:
    print(paper.get("publication_date"), paper.get("title"))
```

A typical discovery loop is: query a reaction, filter to enzymes that are
plausible to work with (cofactor, length, `expression_score`), read the
literature and structures for the survivors, then pick a handful for
experimental validation.

---

## Questions

Open an issue in this repository, or email
[info@dayhofflabs.com](mailto:info@dayhofflabs.com).
