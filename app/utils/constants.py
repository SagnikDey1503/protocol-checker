"""
Application constants and default values.

Centralizing magic numbers and default configurations makes them
easy to find, update, and override via environment variables.
"""

from __future__ import annotations

# ── Pinecone ─────────────────────────────────────────────────

PINECONE_INDEX_NAME = "research-protocol-assistant"
PINECONE_BATCH_SIZE = 100  # Max vectors per upsert batch
PINECONE_NAMESPACES = {
    "protocols": "protocols",
    "memories": "memories",
    "safety": "safety",
    "knowledge": "knowledge",
}

# ── Chunking ─────────────────────────────────────────────────

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
MAX_CHUNK_SIZE = 2048
MIN_CHUNK_SIZE = 100
PARENT_CHUNK_SIZE = 1024
CHILD_CHUNK_SIZE = 256

# ── Retrieval ────────────────────────────────────────────────

DEFAULT_TOP_K = 10
RERANK_TOP_K = 5
MAX_CANDIDATES_FOR_RERANK = 50  # Never rerank more than this
RRF_K = 60  # Reciprocal Rank Fusion constant
MIN_RELEVANCE_THRESHOLD = 0.3  # Minimum score to include in results
CONFIDENCE_HIGH = 0.8
CONFIDENCE_MEDIUM = 0.5
CONFIDENCE_LOW = 0.3

# ── Memory ───────────────────────────────────────────────────

WORKING_MEMORY_TTL = 7200  # 2 hours
EXPERIMENT_STATE_TTL = 86400  # 24 hours
MAX_CONVERSATION_BUFFER = 20
MEMORY_IMPORTANCE_THRESHOLD = 0.4  # Minimum importance to save to long-term memory

# ── Safety ───────────────────────────────────────────────────

# Common hazardous reagents and their risk levels
HAZARDOUS_REAGENTS = {
    "ethidium bromide": {
        "level": "high",
        "warning": "Potent mutagen. Wear nitrile gloves. Dispose in designated EtBr waste.",
        "ppe": ["nitrile gloves", "lab coat", "safety goggles"],
    },
    "formaldehyde": {
        "level": "critical",
        "warning": "Toxic and carcinogenic. Use only in fume hood. Wear appropriate PPE.",
        "ppe": ["nitrile gloves", "lab coat", "safety goggles", "fume hood"],
    },
    "chloroform": {
        "level": "high",
        "warning": "Toxic. Use in fume hood. Avoid inhalation.",
        "ppe": ["nitrile gloves", "lab coat", "fume hood"],
    },
    "phenol": {
        "level": "critical",
        "warning": "Highly corrosive and toxic. Can cause severe burns. Use in fume hood.",
        "ppe": ["nitrile gloves", "lab coat", "safety goggles", "fume hood", "face shield"],
    },
    "acrylamide": {
        "level": "high",
        "warning": "Neurotoxin. Wear gloves. Avoid skin contact and inhalation.",
        "ppe": ["nitrile gloves", "lab coat", "safety goggles"],
    },
    "sodium azide": {
        "level": "critical",
        "warning": "Highly toxic. Explosive with metals. Never pour down drain.",
        "ppe": ["nitrile gloves", "lab coat", "safety goggles", "fume hood"],
    },
    "trizol": {
        "level": "high",
        "warning": "Contains phenol and guanidine isothiocyanate. Use in fume hood.",
        "ppe": ["nitrile gloves", "lab coat", "fume hood"],
    },
    "dmso": {
        "level": "medium",
        "warning": "Rapidly penetrates skin carrying dissolved chemicals. Change gloves immediately if contaminated.",
        "ppe": ["nitrile gloves", "lab coat"],
    },
    "concentrated hcl": {
        "level": "high",
        "warning": "Highly corrosive. Produces toxic fumes. Use in fume hood.",
        "ppe": ["nitrile gloves", "lab coat", "safety goggles", "fume hood"],
    },
    "concentrated naoh": {
        "level": "high",
        "warning": "Highly corrosive. Causes severe burns. Handle with care.",
        "ppe": ["nitrile gloves", "lab coat", "safety goggles"],
    },
    "liquid nitrogen": {
        "level": "critical",
        "warning": "Extreme cold (-196°C). Causes severe frostbite. Use cryogloves and face shield.",
        "ppe": ["cryogloves", "lab coat", "face shield", "safety goggles"],
    },
    "uv light": {
        "level": "medium",
        "warning": "UV exposure can damage eyes and skin. Use UV-blocking goggles or face shield.",
        "ppe": ["uv goggles", "lab coat"],
    },
}

# ── Experiment Types ─────────────────────────────────────────

SUPPORTED_EXPERIMENT_TYPES = [
    "pcr",
    "gel_electrophoresis",
    "dna_extraction",
    "bacterial_transformation",
    "cloning",
    "cell_culture",
    "protein_purification",
    "western_blot",
    "restriction_digest",
    "ligation",
    "miniprep",
    "maxiprep",
    "competent_cell_preparation",
    "colony_pcr",
    "site_directed_mutagenesis",
]

# ── Lab Calculations ─────────────────────────────────────────

# Common buffer concentrations (stock -> working)
COMMON_DILUTIONS = {
    "TAE_buffer": {"stock": "50x", "working": "1x"},
    "TBE_buffer": {"stock": "10x", "working": "0.5x"},
    "PBS": {"stock": "10x", "working": "1x"},
    "loading_dye": {"stock": "6x", "working": "1x"},
}

# Standard PCR parameters
PCR_DEFAULTS = {
    "initial_denaturation": {"temp_c": 95, "duration_s": 300},
    "denaturation": {"temp_c": 95, "duration_s": 30},
    "annealing": {"temp_c": 55, "duration_s": 30},  # Default, should be calculated
    "extension": {"temp_c": 72, "duration_s": 60},  # Per kb of product
    "final_extension": {"temp_c": 72, "duration_s": 600},
    "hold": {"temp_c": 4, "duration_s": None},
    "cycles": 30,
}

# ── File Upload ──────────────────────────────────────────────

ALLOWED_FILE_EXTENSIONS = {".pdf", ".txt", ".doc", ".docx"}
MAX_FILE_SIZE_MB = 50
UPLOAD_DIR = "uploads"

# ── API ──────────────────────────────────────────────────────

API_V1_PREFIX = "/api/v1"
RATE_LIMIT_REQUESTS_PER_MINUTE = 60
