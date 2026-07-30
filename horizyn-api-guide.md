# Horizyn API Guide

The Horizyn API is a hosted service that searches a large library of natural
enzymes for the ones most likely to catalyze a reaction of interest. You supply a
reaction as a SMILES string; the API returns a ranked list of candidate enzymes
with annotations (name, organism, EC numbers, cofactors, sequence length, and
more).

Anyone can get a key — it takes one email round-trip and no account or password.

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

1. [Hosted API vs. this repository](#1-hosted-api-vs-this-repository)
2. [Get an API key](#2-get-an-api-key)
3. [Authenticate](#3-authenticate)
4. [Search for enzymes by reaction](#4-search-for-enzymes-by-reaction)
5. [List available screening sets](#5-list-available-screening-sets)
5a. [List available cofactors](#5a-list-available-cofactors)
6. [Validate SMILES before querying](#6-validate-smiles-before-querying)
7. [Look up an enzyme](#7-look-up-an-enzyme)
8. [Rate limits](#8-rate-limits)
9. [Errors](#9-errors)
10. [Python example](#10-python-example)

---

## 1. Hosted API vs. this repository

The two are complementary; which you want depends on your goal.

| | This repository | Hosted API |
|---|---|---|
| **Enzymes searched** | ~216K bundled protein embeddings | 6.33M proteins |
| **Setup** | Install deps, download ~1GB data + checkpoint | Request a key by email |
| **Hardware** | NVIDIA GPU with 16GB+ VRAM | None — runs server-side |
| **Annotations** | Not included | Name, organism, EC, cofactors, literature, structures |
| **Filtering / clustering** | Not included | EC, transporter class, cofactor, length; EC clustering |
| **Retraining, custom data** | Yes | No |
| **Offline / no rate limit** | Yes | 60 req/min, 10,000 req/month |

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
  address.
- After **5 incorrect** code attempts the code is invalidated — request a new
  one.
- Treat the key like a password. Keep it out of source control; prefer an
  environment variable, as in the examples below.

---

## 3. Authenticate

Send your key as a bearer token on every data request:

```
Authorization: Bearer hzk_live_xxxxxxxx...
```

The two `/keys/*` endpoints above are the only unauthenticated ones. Any other
request without a valid key returns `401 Unauthorized`.

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

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `smiles` | string | — (required) | Reaction SMILES, using `>>` between reactants and products |
| `screening_set` | string | default set | Which enzyme library to search. Omit it to use the default; see [§ 5](#5-list-available-screening-sets) |
| `top_k` | int | `100` | How many nearest enzymes to retrieve before filtering/paging |
| `page` | int | `1` | Page number of results to return |
| `page_size` | int | `20` | Results per page |
| `cluster_by` | string | `null` | Group results by EC number instead of returning a flat list: `"ec3"` (first three EC fields) or `"ec4"` (all four) |
| `cluster_size` | int | `5` | Members shown per cluster when clustering |
| `fetch_metadata` | bool | `true` | Attach enzyme annotations (name, organism, EC, etc.) to results |
| `ec_include` / `ec_exclude` | list[string] | `null` | Keep / drop results by EC number prefix |
| `tc_include` / `tc_exclude` | list[string] | `null` | Keep / drop results by transporter classification |
| `cofactor_include` / `cofactor_exclude` | list[string] | `null` | Keep / drop results by cofactor. Accepts acronyms (`"PLP"`, `"SAM"`, `"TPP"`), element names (`"zinc"`), and ChEBI spellings; case-insensitive |
| `min_length` / `max_length` | int | `null` | Keep results within a sequence-length range |

Note that `top_k` is applied *before* filtering, so aggressive filters combined
with a small `top_k` can return very few rows. Raise `top_k` if you are filtering
hard.

Cofactor filters accept common acronyms and names as well as the ChEBI spellings
used in the annotations, and matching is case-insensitive. All of `"PLP"`,
`"pyridoxal phosphate"` and `"pyridoxal 5'-phosphate"` select the same
PLP-dependent enzymes; `"SAM"`, `"TPP"`, `"PQQ"`, `"B12"`, `"zinc"` and `"Heme"`
work too. A term matching no cofactor in the screening set is reported in the
response's `warnings` rather than silently returning nothing. Call
`GET /cofactors` to list every annotated cofactor with enzyme counts. EC and
transporter-class filters match by prefix instead.

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
  "timings_ms": { "total": 150.0 },
  "quality_warnings": [],
  "warnings": [],
  "metadata_fetched": true
}
```

Results with no usable EC annotation are grouped under an empty cluster key.
Pagination applies to clusters rather than individual enzymes.

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

---

## 6. Validate SMILES before querying

These endpoints check that your input parses, without running a search. They are
useful for giving users fast feedback in an interactive tool, and cheaper than
discovering a typo via a `400` on a full query.

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
  "length": 426
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

---

## 8. Rate limits

Each key is limited to:

- **60 requests per minute**
- **10,000 requests per month**

Exceeding either returns `429 Too Many Requests`, with a `Retry-After` header in
seconds and a body like:

```json
{"error": "rate limited", "retry_after": 42}
```

The service also caps total concurrent queries, so you may briefly receive a
`429` even when under your own limits. In both cases, wait `Retry-After` seconds
and retry.

If your work needs more than these limits, get in touch at
[info@dayhofflabs.com](mailto:info@dayhofflabs.com).

---

## 9. Errors

| Status | Meaning |
|--------|---------|
| `400` | Invalid input (e.g. malformed reaction SMILES). Body includes `quality_warnings`. Also returned for an incorrect verification code. |
| `401` | Missing or invalid API key. |
| `404` | Unknown screening set, or protein not found. |
| `422` | Request body failed schema validation — a missing required field, a wrong type, or an unusable email address on `/keys/*`. The body names the offending field. |
| `429` | Rate limited, or server briefly at capacity — see `Retry-After`. |
| `503` | Service not ready, or verification email temporarily unavailable. Retry shortly. |
| `504` | Query timed out. Retry, optionally with a smaller `top_k`. |
---

## 10. Python example

The API is plain HTTP + JSON, so no SDK is needed. These examples use
[`requests`](https://requests.readthedocs.io/), which is a dependency of this
repository — if you are working outside it, `pip install requests`.

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
        if resp.status_code == 429:
            time.sleep(int(resp.headers.get("Retry-After", 5)))
            continue
        resp.raise_for_status()
        return resp.json()

    raise RuntimeError("still rate limited after 5 attempts")


# PLP-dependent transaminases for the GabT reaction, small enough to be
# practical to express. `top_k` is deliberately large because these filters are
# narrow — see the note in section 4.
data = query_reaction(
    "[NH3+]CCCCCC([O-])=O.[O-]C(=O)CCC(=O)C([O-])=O"
    ">>[O-]C(=O)CCCCC=O.[NH3+][C@@H](CCC([O-])=O)C([O-])=O",
    top_k=1000,
    page_size=10,
    cofactor_include=["PLP"],
    max_length=600,
)

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
