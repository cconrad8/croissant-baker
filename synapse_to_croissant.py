#!/usr/bin/env python3
"""
Generate Croissant metadata for a Synapse entity.

Two modes are supported:

  Full mode (default):
    Downloads all files to a local directory, then runs croissant-maker's
    normal type-inference pipeline. Produces complete Croissant output including
    RecordSet definitions with typed fields for CSV/Parquet/image/WFDB files.

  Metadata-only mode (--no-download):
    Reads entity and file-handle metadata from the Synapse REST API without
    downloading any file content. FileObjects reference Synapse URLs instead of
    local paths. RecordSets with column schemas are generated for:
      * Synapse Table/EntityView entities (column definitions from Synapse API)
      * CSV/TSV files when --add-headers is also passed: streams just the first
        line via an HTTP Range request; all fields typed as sc:Text.
    SHA-256 checksums are omitted (Synapse stores MD5, not SHA-256).

Requirements:
    pip install synapseclient
    (croissant-maker must also be installed in the same Python environment)

Authentication:
    Uses ~/.synapseConfig or the SYNAPSE_AUTH_TOKEN environment variable.
    See: https://python-docs.synapse.org/en/stable/tutorials/authentication/

Usage:
    python synapse_to_croissant.py syn52623570
    python synapse_to_croissant.py syn52623570 --no-download
    python synapse_to_croissant.py syn52623570 --no-download --add-headers
    python synapse_to_croissant.py syn52623570.4
    python synapse_to_croissant.py "https://www.synapse.org/Synapse:syn52623570.4/datasets/"
    python synapse_to_croissant.py syn52623570 --creator "Jane Doe,jane@example.com"
    python synapse_to_croissant.py syn52623570 --download-dir ./data --keep-files
"""

import argparse
import json
import mimetypes
import re
import shutil
import sys
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import mlcroissant as mlc
import synapseclient
import synapseutils
from synapseclient.models import Dataset as SynapseDataset
from synapseclient.models import Folder, Project

from croissant_maker.handlers.utils import sanitize_id
from croissant_maker.metadata_generator import MetadataGenerator

# ---------------------------------------------------------------------------
# MIME-type inference
# ---------------------------------------------------------------------------

# Bioinformatics / life-sciences extensions not in Python's mimetypes database.
# Keys are lower-cased suffixes (may be compound, e.g. ".fastq.gz").
_BIO_MIME: dict[str, str] = {
    # Sequencing reads
    ".fastq": "text/x-fastq",
    ".fq": "text/x-fastq",
    ".fastq.gz": "application/gzip",
    ".fq.gz": "application/gzip",
    # Alignments
    ".bam": "application/x-bam",
    ".bai": "application/x-bam-index",
    ".cram": "application/x-cram",
    ".crai": "application/x-cram-index",
    ".sam": "text/x-sam",
    # Variant calls
    ".vcf": "text/x-vcf",
    ".vcf.gz": "application/gzip",
    ".bcf": "application/x-bcf",
    # Genome annotation / intervals
    ".bed": "text/x-bed",
    ".bed.gz": "application/gzip",
    ".gtf": "text/x-gtf",
    ".gtf.gz": "application/gzip",
    ".gff": "text/x-gff",
    ".gff3": "text/x-gff",
    # Quantification / expression
    ".sf": "text/tab-separated-values",  # Salmon quant.sf
    ".quant": "text/tab-separated-values",
    ".counts": "text/tab-separated-values",
    # Single-cell / HDF5 containers
    ".h5ad": "application/x-hdf5",
    ".h5": "application/x-hdf5",
    ".hdf5": "application/x-hdf5",
    ".loom": "application/x-hdf5",
    # Flow cytometry
    ".fcs": "application/vnd.isac.fcs",
    # R data
    ".rds": "application/x-r-rds",
    ".rda": "application/x-r-data",
    ".rdata": "application/x-r-data",
    # Tabular
    ".tsv": "text/tab-separated-values",
    ".tsv.gz": "application/gzip",
    ".csv.gz": "application/gzip",
    # Images / whole-slide
    ".svs": "image/vnd.svs",
    ".ndpi": "image/x-ndpi",
    ".ome.tiff": "image/ome-tiff",
    ".ome.tif": "image/ome-tiff",
    # Array / methylation
    ".idat": "application/x-idat",
    ".cel": "application/x-cel",
    # Misc
    ".maf": "text/x-maf",
    ".plink": "application/x-plink",
    ".log": "text/plain",
    ".md": "text/markdown",
}


def _mime_from_name(name: str, synapse_content_type: str | None) -> str:
    """Return the best MIME type for *name*.

    Priority:
      1. Bio-specific map (compound suffix first, then simple suffix)
      2. Python's ``mimetypes`` database
      3. Synapse's ``contentType`` (if not generic octet-stream)
      4. ``"application/octet-stream"`` as last resort
    """
    nl = name.lower()

    # Compound suffixes (.fastq.gz, .vcf.gz, …)
    for suffix, mime in _BIO_MIME.items():
        if len(suffix) > 4 and nl.endswith(suffix):
            return mime

    # Simple suffixes
    for suffix, mime in _BIO_MIME.items():
        if len(suffix) <= 4 and nl.endswith(suffix):
            return mime

    # Python stdlib mimetypes
    stdlib_mime, _ = mimetypes.guess_type(name)
    if stdlib_mime:
        return stdlib_mime

    # Synapse content type (skip generic fallback)
    if synapse_content_type and synapse_content_type != "application/octet-stream":
        return synapse_content_type

    return "application/octet-stream"


# ---------------------------------------------------------------------------
# Synapse concrete-type constants
# ---------------------------------------------------------------------------
_FILE_TYPE = "org.sagebionetworks.repo.model.FileEntity"
_FOLDER_TYPE = "org.sagebionetworks.repo.model.Folder"
_PROJECT_TYPE = "org.sagebionetworks.repo.model.Project"
_TABLE_TYPE = "org.sagebionetworks.repo.model.table.TableEntity"
_ENTITY_VIEW_TYPE = "org.sagebionetworks.repo.model.table.EntityView"
_DATASET_TYPE = "org.sagebionetworks.repo.model.table.Dataset"
_DATASET_COLLECTION_TYPE = "org.sagebionetworks.repo.model.table.DatasetCollection"

_TABLE_TYPES = {_TABLE_TYPE, _ENTITY_VIEW_TYPE}
_DATASET_TYPES = {_DATASET_TYPE, _DATASET_COLLECTION_TYPE}

# ---------------------------------------------------------------------------
# Synapse column type → schema.org / Croissant data type
# ---------------------------------------------------------------------------
_SYNAPSE_TYPE_MAP: dict[str, str] = {
    "STRING": "sc:Text",
    "LARGETEXT": "sc:Text",
    "MEDIUMTEXT": "sc:Text",
    "LINK": "sc:URL",
    "INTEGER": "sc:Integer",
    "DOUBLE": "sc:Float",
    "BOOLEAN": "sc:Boolean",
    "DATE": "sc:Date",
    "FILEHANDLEID": "sc:URL",
    "ENTITYID": "sc:URL",
    "USERID": "sc:Integer",
    "SUBMISSIONID": "sc:Integer",
    "JSON": "sc:Text",
    "STRING_LIST": "sc:Text",
    "INTEGER_LIST": "sc:Integer",
    "BOOLEAN_LIST": "sc:Boolean",
    "DATE_LIST": "sc:Date",
    "ENTITYID_LIST": "sc:URL",
    "USERID_LIST": "sc:Integer",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_synapse_id(id_or_url: str) -> tuple[str, str | None]:
    """Extract Synapse ID and optional version from an ID, versioned ID, or URL.

    Examples:
        "syn52623570"       -> ("syn52623570", None)
        "syn52623570.4"     -> ("syn52623570", "4")
        "https://...syn52623570.4/datasets/" -> ("syn52623570", "4")
    """
    match = re.search(r"\b(syn\d+)(?:\.(\d+))?\b", id_or_url, re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not find a Synapse ID in: {id_or_url!r}")
    return match.group(1), match.group(2)


def _parse_creators(creator_args: list[str] | None) -> list[dict] | None:
    """Convert --creator CLI strings into creator dicts for MetadataGenerator."""
    if not creator_args:
        return None
    creators = []
    for s in creator_args:
        parts = [p.strip() for p in s.split(",")]
        if not parts[0]:
            continue
        c: dict[str, str] = {"name": parts[0]}
        if len(parts) > 1 and parts[1]:
            c["email"] = parts[1]
        if len(parts) > 2 and parts[2]:
            c["url"] = parts[2]
        creators.append(c)
    return creators or None


def _get_annotation_str(annotations: dict, *keys: str) -> str | None:
    """Return the first non-empty string found for any of the given annotation keys."""
    for key in keys:
        val = annotations.get(key)
        if isinstance(val, list):
            val = val[0] if val else None
        if val:
            return str(val)
    return None


def _get_annotation_list(annotations: dict, *keys: str) -> list[str]:
    """Return all values for the first matching key as a flat list of strings."""
    for key in keys:
        val = annotations.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            return [str(v) for v in val if v]
        return [str(val)]
    return []


def _resolve_creators_from_synapse(
    syn: synapseclient.Synapse,
    raw_entity: dict,
    annotations: dict,
) -> list[dict] | None:
    """Infer creator info from annotations, falling back to the entity owner's profile.

    Checks these annotation keys (case-sensitive, first match wins):
      name  : creator, Creator, PI, principalInvestigator, contact, dataContributor,
               authors, author, investigator, Investigator
      email : creatorEmail, contactEmail, email, Email
      ORCID : orcid, ORCID

    If no creator annotation is found, looks up the Synapse user profile of
    ``raw_entity["createdBy"]`` and constructs a creator from their name/email.
    Returns None only if all strategies fail.
    """
    # --- Annotations first -------------------------------------------------------
    name = _get_annotation_str(
        annotations,
        "creator",
        "Creator",
        "PI",
        "principalInvestigator",
        "investigator",
        "Investigator",
        "contact",
        "dataContributor",
        "authors",
        "author",
    )
    if name:
        email = _get_annotation_str(
            annotations, "creatorEmail", "contactEmail", "email", "Email"
        )
        orcid = _get_annotation_str(annotations, "orcid", "ORCID")
        c: dict[str, str] = {"name": name}
        if email:
            c["email"] = email
        if orcid:
            orcid_id = orcid.replace("https://orcid.org/", "").strip("/")
            c["url"] = f"https://orcid.org/{orcid_id}"
        return [c]

    # --- Synapse user-profile fallback -------------------------------------------
    created_by = raw_entity.get("createdBy")
    if not created_by:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            profile = syn.getUserProfile(created_by)
        first = profile.get("firstName") or ""
        last = profile.get("lastName") or ""
        username = profile.get("userName") or ""
        full_name = f"{first} {last}".strip() or username or f"Synapse:{created_by}"
        emails: list = profile.get("emails") or []
        c = {
            "name": full_name,
            "url": f"https://www.synapse.org/#!Profile:{created_by}",
        }
        if emails:
            c["email"] = emails[0]
        print(f"  Creator : {full_name} (resolved from Synapse profile)")
        return [c]
    except Exception as e:
        print(
            f"  Note: Could not resolve creator from Synapse profile: {e}",
            file=sys.stderr,
        )
        return None


# MIME types that represent tabular data we can generate RecordSets from.
_TABULAR_MIMES = {"text/csv", "text/tab-separated-values"}


def _stream_tabular_header(
    syn: synapseclient.Synapse,
    entity_ref: str,
    content_type: str,
) -> list[str] | None:
    """Return the column names of a tabular file by fetching only its first line.

    Obtains a Synapse pre-signed URL then issues an HTTP Range request for the
    first 8 KiB so that the full file is never downloaded.  Returns a list of
    column name strings, or None when:

    * ``content_type`` is not in ``_TABULAR_MIMES``
    * the request fails for any reason
    * the first line is empty or contains no parseable columns

    Column types are not inferred — every Field receives ``sc:Text``.  For full
    type inference, run without ``--no-download``.
    """
    if content_type not in _TABULAR_MIMES:
        return None

    import csv
    import urllib.request

    # Build the correct REST path for versioned vs. unversioned refs
    if "." in entity_ref:
        syn_id, ver = entity_ref.split(".", 1)
        rest_path = f"/entity/{syn_id}/version/{ver}/file?redirect=false"
    else:
        rest_path = f"/entity/{entity_ref}/file?redirect=false"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            url = syn.restGET(rest_path)

        req = urllib.request.Request(url, headers={"Range": "bytes=0-8191"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            chunk = resp.read().decode("utf-8", errors="replace")

        first_line = chunk.splitlines()[0].rstrip("\r\n")
        if content_type == "text/csv":
            columns = next(csv.reader([first_line]))
        else:  # TSV / quant.sf / counts / etc.
            columns = first_line.split("\t")

        columns = [c.strip() for c in columns if c.strip()]
        return columns or None

    except Exception as e:
        print(
            f"  Note: Could not stream header for {entity_ref}: {e}",
            file=sys.stderr,
        )
        return None


def _get_file_handle_info(syn: synapseclient.Synapse, entity_ref: str) -> dict:
    """Return file-handle metadata for a FileEntity without downloading its content.

    Uses the legacy ``syn.get(downloadFile=False)`` path because the new models API
    (``File.get()``) always downloads the file content.  The deprecation warning is
    suppressed until a metadata-only path exists in the new API.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        ent = syn.get(entity_ref, downloadFile=False)
    fh: dict = ent._file_handle or {}
    name = fh.get("fileName") or ent.name
    return {
        "name": name,
        "content_size": str(fh.get("contentSize")) if fh.get("contentSize") else None,
        "content_type": _mime_from_name(name, fh.get("contentType")),
        "md5": fh.get("contentMd5"),
        "synapse_id": entity_ref,
    }


def _fetch_file_infos_concurrent(
    syn: synapseclient.Synapse,
    refs: list[str],
    max_workers: int = 8,
) -> list[dict]:
    """Fetch file-handle metadata for *refs* in parallel.

    Each ref is a Synapse entity ID (optionally versioned, e.g. ``syn123.2``).
    Results are returned in the same order as *refs*; failed refs are skipped
    with a warning printed to stderr.
    """
    results: list[dict | None] = [None] * len(refs)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(_get_file_handle_info, syn, ref): i
            for i, ref in enumerate(refs)
        }
        done = 0
        total = len(refs)
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            done += 1
            try:
                results[idx] = future.result()
            except Exception as e:
                print(
                    f"  Warning: Could not get info for {refs[idx]}: {e}",
                    file=sys.stderr,
                )
            # Simple in-place progress (overwrites the same line)
            print(f"  Fetching metadata: {done}/{total}", end="\r", flush=True)
    print()  # newline after progress line
    return [r for r in results if r is not None]


def _iter_dataset_items(
    syn: synapseclient.Synapse,
    synapse_id: str,
    version: str | None,
    entity_version: int | None = None,
    max_workers: int = 8,
) -> list[dict]:
    """Return file-handle info for every item in a Synapse Dataset.

    Strategy (each step tried in order):
      1. ``Dataset(id=...).get()``  – models API, preserves pinned file versions
      2. ``synapseutils.walk()``    – generic container walk (no pinned versions)

    File-handle metadata is fetched concurrently (``max_workers`` threads).
    """
    # --- Strategy 1: models API ------------------------------------------------
    try:
        ds = SynapseDataset(id=synapse_id)
        ds.get(synapse_client=syn)
        items = ds.items or []
        refs = [
            f"{item.id}.{item.version}" if getattr(item, "version", None) else item.id
            for item in items
        ]
        return _fetch_file_infos_concurrent(syn, refs, max_workers=max_workers)
    except Exception as e:
        print(
            f"  Note: Dataset models API failed ({e}), falling back to walk...",
            file=sys.stderr,
        )

    # --- Strategy 2: synapseutils.walk() ---------------------------------------
    print(
        "  Note: using synapseutils.walk() (pinned versions not preserved).",
        file=sys.stderr,
    )
    refs = []
    for _dirpath, _dirnames, files in synapseutils.walk(syn, synapse_id):
        for _fname, fid in files:
            refs.append(fid)
    return _fetch_file_infos_concurrent(syn, refs, max_workers=max_workers)


def _walk_to_file_infos(
    syn: synapseclient.Synapse,
    synapse_id: str,
    version: str | None,
    concrete_type: str,
    entity_version: int | None = None,
    max_workers: int = 8,
) -> list[dict]:
    """Walk an entity tree and return a flat list of file-info dicts.

    Table/EntityView entities are represented as a single entry with
    ``is_table=True`` so the caller can fetch their column schema separately.
    """
    entity_ref = f"{synapse_id}.{version}" if version else synapse_id

    if concrete_type == _FILE_TYPE:
        return [_get_file_handle_info(syn, entity_ref)]

    if concrete_type in _DATASET_TYPES:
        return _iter_dataset_items(
            syn,
            synapse_id,
            version,
            entity_version=entity_version,
            max_workers=max_workers,
        )

    if concrete_type in _TABLE_TYPES:
        raw = syn.restGET(f"/entity/{entity_ref}")
        return [
            {
                "name": raw.get("name", synapse_id),
                "synapse_id": synapse_id,
                "is_table": True,
            }
        ]

    if concrete_type in {_FOLDER_TYPE, _PROJECT_TYPE}:
        refs = []
        for _dirpath, _dirnames, files in synapseutils.walk(syn, synapse_id):
            for _fname, fid in files:
                refs.append(fid)
        return _fetch_file_infos_concurrent(syn, refs, max_workers=max_workers)

    # Unknown type: attempt folder walk, then dataset items as fallback
    print(
        f"  Warning: Unknown entity type '{concrete_type.split('.')[-1]}', "
        "attempting folder walk...",
        file=sys.stderr,
    )
    try:
        refs = []
        for _dirpath, _dirnames, files in synapseutils.walk(syn, synapse_id):
            for _fname, fid in files:
                refs.append(fid)
        return _fetch_file_infos_concurrent(syn, refs, max_workers=max_workers)
    except Exception:
        return _iter_dataset_items(syn, synapse_id, version, max_workers=max_workers)


def _get_table_columns(syn: synapseclient.Synapse, table_id: str) -> dict[str, str]:
    """Return {column_name: croissant_type} for a Synapse Table or EntityView."""
    cols: dict[str, str] = {}
    try:
        for col in syn.getColumns(table_id):
            croissant_type = _SYNAPSE_TYPE_MAP.get(
                col.get("columnType", "STRING"), "sc:Text"
            )
            cols[col["name"]] = croissant_type
    except Exception as e:
        print(
            f"  Warning: Could not retrieve columns for {table_id}: {e}",
            file=sys.stderr,
        )
    return cols


# ---------------------------------------------------------------------------
# Mode A: metadata-only (no download)
# ---------------------------------------------------------------------------


def _metadata_only(
    syn: synapseclient.Synapse,
    synapse_id: str,
    version: str | None,
    raw_entity: dict,
    *,
    description: str | None,
    license_value: str | None,
    parsed_creators: list[dict] | None,
    dataset_version: str | None,
    dataset_url: str,
    output_path: str,
    validate: bool,
    entity_version: int | None = None,
    max_workers: int = 8,
    doi: str | None = None,
    keywords: list[str] | None = None,
    add_headers: bool = False,
) -> dict:
    """Build Croissant metadata using only the Synapse API (no file downloads).

    FileObjects point to Synapse URLs. RecordSets with typed fields are created
    for two kinds of entities:

    * **Synapse Table / EntityView** – column schema is fetched from the Synapse
      API; types are mapped to Croissant types.
    * **Tabular file** (CSV / TSV / quant.sf / …) when ``add_headers=True`` –
      the first 8 KiB of the file is streamed via a Range request so that column
      names can be extracted without a full download.  All fields receive
      ``sc:Text``; for precise type inference use the full download mode.

    SHA-256 checksums are omitted because Synapse stores MD5, not SHA-256.
    """
    concrete_type = raw_entity.get("concreteType", "")
    dataset_name = raw_entity.get("name") or synapse_id

    print("Walking entity tree (no download)...")
    file_infos = _walk_to_file_infos(
        syn,
        synapse_id,
        version,
        concrete_type,
        entity_version=entity_version,
        max_workers=max_workers,
    )
    if not file_infos:
        raise ValueError("No file entities found under this Synapse entity.")
    print(f"  Found {len(file_infos)} file entities.")

    # For Dataset entities, also inject a sentinel so the loop below can
    # generate a RecordSet from the dataset's column schema (the metadata
    # annotations that describe each item in the dataset).
    if concrete_type in _DATASET_TYPES:
        file_infos = list(file_infos) + [
            {"name": dataset_name, "synapse_id": synapse_id, "is_dataset_schema": True}
        ]

    # Build mlcroissant metadata object
    if parsed_creators:
        creator_objects = [mlc.Person(**c) for c in parsed_creators]
    else:
        creator_objects = [
            mlc.Person(name="Dataset Creator", email="creator@example.com")
        ]

    # Build cite_as: prefer DOI, fall back to Synapse URL
    if doi:
        doi_url = doi if doi.startswith("http") else f"https://doi.org/{doi}"
        cite_as_str = f"{dataset_name}. {doi_url}"
    else:
        cite_as_str = f"{dataset_name}. Synapse ID: {synapse_id}. {dataset_url}"

    metadata_kwargs: dict = dict(
        name=dataset_name,
        description=description
        or f"Dataset '{dataset_name}' from Synapse ({synapse_id})",
        url=dataset_url,
        license=license_value or "https://creativecommons.org/licenses/by/4.0/",
        creators=creator_objects,
        date_published=datetime.now(),
        version=dataset_version or "1.0.0",
        cite_as=cite_as_str,
    )
    if keywords:
        metadata_kwargs["keywords"] = keywords

    metadata = mlc.Metadata(**metadata_kwargs)

    file_objects: list = []
    record_sets: list = []

    for i, info in enumerate(file_infos):
        file_id_str = f"file_{i}"
        syn_url = f"https://www.synapse.org/Synapse:{info['synapse_id']}"

        if info.get("is_table"):
            # Synapse Table/EntityView: fetch column schema and build RecordSet
            table_syn_id = info["synapse_id"]
            table_name = info["name"]

            file_obj = mlc.FileObject(
                id=file_id_str,
                name=table_name,
                description=f"Synapse Table {table_syn_id}",
                content_url=syn_url,
                encoding_formats=["text/csv"],
            )
            file_objects.append(file_obj)

            columns = _get_table_columns(syn, table_syn_id)
            rs_id = sanitize_id(table_name)
            fields = [
                mlc.Field(
                    id=f"{rs_id}/{sanitize_id(col_name)}",
                    name=col_name,
                    description=f"Column '{col_name}'",
                    data_types=[col_type],
                    source=mlc.Source(
                        id=f"{rs_id}/{sanitize_id(col_name)}/source",
                        file_object=file_id_str,
                        extract=mlc.Extract(column=col_name),
                    ),
                )
                for col_name, col_type in columns.items()
            ]
            if fields:
                record_sets.append(
                    mlc.RecordSet(
                        id=rs_id,
                        name=table_name,
                        description=f"Synapse Table {table_syn_id}: {len(fields)} columns",
                        fields=fields,
                    )
                )
        elif info.get("is_dataset_schema"):
            # Synapse Dataset: fetch the column schema (item-level metadata
            # annotations) and generate a RecordSet from it — no download needed.
            dataset_syn_id = info["synapse_id"]
            dataset_schema_name = info["name"]

            file_obj = mlc.FileObject(
                id=file_id_str,
                name=dataset_schema_name,
                description=f"Synapse Dataset metadata schema for {dataset_syn_id}",
                content_url=syn_url,
                encoding_formats=["text/csv"],
            )
            file_objects.append(file_obj)

            columns = _get_table_columns(syn, dataset_syn_id)
            rs_id = sanitize_id(f"{dataset_schema_name}-schema")
            fields = [
                mlc.Field(
                    id=f"{rs_id}/{sanitize_id(col_name)}",
                    name=col_name,
                    description=f"Column '{col_name}'",
                    data_types=[col_type],
                    source=mlc.Source(
                        id=f"{rs_id}/{sanitize_id(col_name)}/source",
                        file_object=file_id_str,
                        extract=mlc.Extract(column=col_name),
                    ),
                )
                for col_name, col_type in columns.items()
            ]
            if fields:
                record_sets.append(
                    mlc.RecordSet(
                        id=rs_id,
                        name=f"{dataset_schema_name} (metadata schema)",
                        description=(
                            f"Item-level metadata schema for Synapse Dataset "
                            f"{dataset_syn_id}: {len(fields)} columns"
                        ),
                        fields=fields,
                    )
                )

        else:
            # Regular file entity
            content_type = info.get("content_type", "application/octet-stream")
            file_obj = mlc.FileObject(
                id=file_id_str,
                name=info["name"],
                content_url=syn_url,
                encoding_formats=[content_type],
                content_size=info.get("content_size"),
                md5=info.get(
                    "md5"
                ),  # Synapse stores MD5; satisfies the md5/sha256 requirement
            )
            file_objects.append(file_obj)

            # Optionally stream the first line to build a RecordSet for tabular files
            if add_headers and content_type in _TABULAR_MIMES:
                columns = _stream_tabular_header(syn, info["synapse_id"], content_type)
                if columns:
                    rs_id = sanitize_id(info["name"])
                    fields = [
                        mlc.Field(
                            id=f"{rs_id}/{sanitize_id(col)}",
                            name=col,
                            description=f"Column '{col}'",
                            data_types=["sc:Text"],
                            source=mlc.Source(
                                id=f"{rs_id}/{sanitize_id(col)}/source",
                                file_object=file_id_str,
                                extract=mlc.Extract(column=col),
                            ),
                        )
                        for col in columns
                    ]
                    record_sets.append(
                        mlc.RecordSet(
                            id=rs_id,
                            name=info["name"],
                            description=(
                                f"{len(fields)} columns (header only; "
                                "types inferred as sc:Text)"
                            ),
                            fields=fields,
                        )
                    )

    metadata.distribution = file_objects
    metadata.record_sets = record_sets

    metadata_dict = metadata.to_json()
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Always write the file first — validation is advisory and must never block the save.
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
        f.write("\n")

    if validate:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonld", delete=False
            ) as tmp:
                json.dump(metadata_dict, tmp, indent=2, ensure_ascii=False)
                tmp_path = tmp.name
            mlc.Dataset(tmp_path)
            print("  Validation: OK")
        except Exception as e:
            print(
                f"  Warning: mlcroissant validation reported issues (file was saved anyway):\n"
                f"    {e}",
                file=sys.stderr,
            )
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    return metadata_dict


# ---------------------------------------------------------------------------
# Mode B: full download + type inference
# ---------------------------------------------------------------------------


def _full_download(
    syn: synapseclient.Synapse,
    synapse_id: str,
    version: str | None,
    raw_entity: dict,
    *,
    download_dir: Path,
    description: str | None,
    license_value: str | None,
    parsed_creators: list[dict] | None,
    dataset_version: str | None,
    dataset_url: str,
    output_path: str,
    validate: bool,
    count_csv_rows: bool,
    max_workers: int = 8,
) -> dict:
    """Download all entity files then run MetadataGenerator for full type inference."""
    concrete_type = raw_entity.get("concreteType", "")
    entity_ref = f"{synapse_id}.{version}" if version else synapse_id
    dataset_name = raw_entity.get("name") or synapse_id

    print(f"Downloading files to {download_dir} ...")

    if concrete_type == _FILE_TYPE:
        f = syn.get(
            entity_ref,
            downloadLocation=str(download_dir),
            ifcollision="overwrite.local",
        )
        print(f"  Downloaded: {Path(f.path).name}")

    elif concrete_type == _FOLDER_TYPE:
        Folder(id=synapse_id).sync_from_synapse(
            path=str(download_dir), if_collision="overwrite.local"
        )

    elif concrete_type == _PROJECT_TYPE:
        Project(id=synapse_id).sync_from_synapse(
            path=str(download_dir), if_collision="overwrite.local"
        )

    elif concrete_type in _DATASET_TYPES:
        entity_version = raw_entity.get("versionNumber")
        infos = _iter_dataset_items(
            syn,
            synapse_id,
            version,
            entity_version=entity_version,
            max_workers=max_workers,
        )
        downloaded = 0

        def _download_one(ref: str) -> str:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                f = syn.get(
                    ref,
                    downloadLocation=str(download_dir),
                    ifcollision="overwrite.local",
                )
            return Path(f.path).name

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_download_one, info["synapse_id"]): info for info in infos
            }
            for future in as_completed(futures):
                try:
                    name = future.result()
                    downloaded += 1
                    print(f"  Downloaded: {name}")
                except Exception as e:
                    print(f"  Warning: Download failed: {e}", file=sys.stderr)
        print(f"  Total: {downloaded} files.")

    else:
        print(
            f"  Warning: Unknown type '{concrete_type.split('.')[-1]}', "
            "attempting folder sync...",
            file=sys.stderr,
        )
        try:
            Folder(id=synapse_id).sync_from_synapse(
                path=str(download_dir), if_collision="overwrite.local"
            )
        except Exception:
            raise ValueError(f"Cannot download entity of type: {concrete_type}")

    file_count = sum(1 for p in download_dir.rglob("*") if p.is_file())
    if file_count == 0:
        raise ValueError("No files were downloaded from Synapse.")
    print(f"  Total local files: {file_count}")

    print("\nGenerating Croissant metadata...")
    generator = MetadataGenerator(
        dataset_path=str(download_dir),
        name=dataset_name,
        description=description,
        url=dataset_url,
        license=license_value,
        version=dataset_version,
        creators=parsed_creators,
        count_csv_rows=count_csv_rows,
    )
    generator.save_metadata(output_path, validate=validate)
    return generator.generate_metadata()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Croissant metadata for a Synapse entity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Full mode — downloads files, infers column schemas:
  python synapse_to_croissant.py syn52623570

  # Metadata-only mode — no download, uses Synapse API metadata:
  python synapse_to_croissant.py syn52623570 --no-download

  # Metadata-only + stream CSV/TSV headers for RecordSets (no full download):
  python synapse_to_croissant.py syn52623570 --no-download --add-headers

  # Versioned ID or full URL:
  python synapse_to_croissant.py syn52623570.4
  python synapse_to_croissant.py "https://www.synapse.org/Synapse:syn52623570.4/datasets/"

  # Supply creator / license:
  python synapse_to_croissant.py syn52623570 \\
      --creator "Jane Doe,jane@example.com,https://orcid.org/0000-0000-0000-0000" \\
      --license CC-BY-4.0

  # Keep downloaded files after run:
  python synapse_to_croissant.py syn52623570 --download-dir ./data --keep-files
        """,
    )
    parser.add_argument(
        "synapse_id",
        help="Synapse ID (e.g. syn52623570), versioned ID (syn52623570.4), or Synapse URL",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output .jsonld path (default: <synapse_id>-croissant.jsonld)",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help=(
            "Metadata-only mode: build Croissant FileObjects from the Synapse API without "
            "downloading files. URLs point to Synapse. Column schemas are only available "
            "for Synapse Table/EntityView entities (or tabular files with --add-headers)."
        ),
    )
    parser.add_argument(
        "--add-headers",
        action="store_true",
        help=(
            "With --no-download: stream the first line of each CSV/TSV file via a "
            "Range request and generate a RecordSet with column names. "
            "All fields are typed sc:Text (no full type inference). "
            "Ignored in full download mode."
        ),
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip mlcroissant validation of the generated metadata",
    )
    parser.add_argument(
        "--download-dir",
        help=(
            "Directory for downloaded files (default: a temp directory). "
            "Implies --keep-files."
        ),
    )
    parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Keep downloaded files after generating metadata (full mode only)",
    )
    parser.add_argument(
        "--creator",
        action="append",
        dest="creators",
        metavar="NAME[,EMAIL[,URL]]",
        help=(
            "Creator info in 'Name[,Email[,URL]]' format. "
            "Repeat for multiple creators. Overrides Synapse metadata."
        ),
    )
    parser.add_argument(
        "--description",
        help="Dataset description (overrides Synapse entity description)",
    )
    parser.add_argument(
        "--license",
        help="License URL or SPDX identifier (e.g. CC-BY-4.0). Overrides Synapse annotations.",
    )
    parser.add_argument(
        "--count-csv-rows",
        action="store_true",
        help="Count exact row numbers for CSV files (full mode only; slow for large datasets)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        metavar="N",
        help=(
            "Number of parallel threads for fetching file metadata / downloading files "
            "(default: 8). Increase for large datasets with many files."
        ),
    )
    args = parser.parse_args()

    # Parse Synapse ID
    try:
        synapse_id, version = parse_synapse_id(args.synapse_id)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    entity_ref = f"{synapse_id}.{version}" if version else synapse_id
    output_path = args.output or f"croissant/{synapse_id}-croissant.jsonld"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Login
    print("Logging in to Synapse (uses ~/.synapseConfig or SYNAPSE_AUTH_TOKEN)...")
    try:
        syn = synapseclient.login(silent=True)
    except Exception as e:
        print(f"Error: Synapse login failed: {e}", file=sys.stderr)
        print(
            "Configure ~/.synapseConfig or set the SYNAPSE_AUTH_TOKEN environment variable.\n"
            "See: https://python-docs.synapse.org/en/stable/tutorials/authentication/",
            file=sys.stderr,
        )
        sys.exit(1)

    # Fetch entity metadata
    print(f"Fetching entity info for {entity_ref}...")
    try:
        raw_entity = syn.restGET(f"/entity/{entity_ref}")
    except Exception as e:
        print(f"Error: Could not fetch entity {entity_ref}: {e}", file=sys.stderr)
        sys.exit(1)

    concrete_type = raw_entity.get("concreteType", "")
    entity_name = raw_entity.get("name") or synapse_id
    entity_desc = raw_entity.get("description") or raw_entity.get("comments")
    print(f"  Name : {entity_name}")
    print(f"  Type : {concrete_type.split('.')[-1]}")

    # Pull annotations for license, etc.
    # get_annotations() is deprecated in ≥4.9.0 but still works; suppress the warning.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            annotations = syn.get_annotations(entity_ref)
    except Exception:
        annotations = {}

    # Resolve final metadata values
    description = args.description or entity_desc or None
    license_value = args.license or _get_annotation_str(
        annotations, "license", "License"
    )
    doi = _get_annotation_str(annotations, "doi", "DOI", "digitalObjectIdentifier")
    keywords = (
        _get_annotation_list(
            annotations, "keywords", "keyword", "tags", "tag", "Keywords"
        )
        or None
    )
    dataset_version = (
        str(version)
        if version
        else (
            str(raw_entity["versionNumber"])
            if raw_entity.get("versionNumber")
            else None
        )
    )
    dataset_url = f"https://www.synapse.org/Synapse:{synapse_id}"

    # Creator: CLI flag > annotations > Synapse user profile
    parsed_creators = _parse_creators(args.creators) or _resolve_creators_from_synapse(
        syn, raw_entity, annotations
    )
    validate = not args.no_validate

    try:
        if args.no_download:
            print("\nMode: metadata-only (no file download)")
            metadata_dict = _metadata_only(
                syn,
                synapse_id,
                version,
                raw_entity,
                description=description,
                license_value=license_value,
                parsed_creators=parsed_creators,
                dataset_version=dataset_version,
                dataset_url=dataset_url,
                output_path=output_path,
                validate=validate,
                entity_version=raw_entity.get("versionNumber"),
                max_workers=args.workers,
                doi=doi,
                keywords=keywords,
                add_headers=args.add_headers,
            )
        else:
            print("\nMode: full (downloading files for schema inference)")
            cleanup_dir = False
            if args.download_dir:
                download_dir = Path(args.download_dir)
                download_dir.mkdir(parents=True, exist_ok=True)
            else:
                tmp = tempfile.mkdtemp(prefix=f"synapse_{synapse_id}_")
                download_dir = Path(tmp)
                cleanup_dir = not args.keep_files

            try:
                metadata_dict = _full_download(
                    syn,
                    synapse_id,
                    version,
                    raw_entity,
                    download_dir=download_dir,
                    description=description,
                    license_value=license_value,
                    parsed_creators=parsed_creators,
                    dataset_version=dataset_version,
                    dataset_url=dataset_url,
                    output_path=output_path,
                    validate=validate,
                    count_csv_rows=args.count_csv_rows,
                    max_workers=args.workers,
                )
            finally:
                if cleanup_dir:
                    shutil.rmtree(download_dir, ignore_errors=True)
                    print("Cleaned up temporary download directory.")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)

    # Summary
    file_count = len(metadata_dict.get("distribution", []))
    record_count = len(metadata_dict.get("recordSet", []))
    print("\nSuccess!")
    print(f"  Files       : {file_count}")
    print(f"  Record sets : {record_count}")
    print(f"  Output      : {output_path}")

    if not validate:
        print(f"\nTip: Run `croissant-maker validate {output_path}` to validate later.")

    missing = []
    if not parsed_creators:
        missing.append("--creator")
    if not description:
        missing.append("--description")
    if not license_value:
        missing.append("--license")
    if missing:
        print(
            f"\nNote: {', '.join(missing)} not provided; defaults were used. "
            "Review the output before publishing.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
