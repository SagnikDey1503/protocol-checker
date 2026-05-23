"""
Seed common synthetic biology protocols into the system.

Pre-loads a library of beginner-friendly protocols for
PCR, gel electrophoresis, DNA extraction, bacterial transformation,
cloning, and cell culture.

Run: python -m scripts.seed_protocols
"""

import asyncio
import logging
import sys
import uuid
from datetime import datetime

sys.path.insert(0, ".")

from app.config import get_settings
from app.core.embeddings import embed_texts
from app.utils.constants import PINECONE_BATCH_SIZE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ── Seed Protocol Data ───────────────────────────────────────

SEED_PROTOCOLS = [
    {
        "title": "Standard PCR Protocol",
        "experiment_type": "pcr",
        "steps": [
            {
                "step_number": 1,
                "title": "Prepare Reaction Mix",
                "description": "In a PCR tube on ice, combine the following components: 25 µL 2x PCR Master Mix, 2 µL forward primer (10 µM), 2 µL reverse primer (10 µM), 1-5 µL template DNA (10-100 ng), and nuclease-free water to 50 µL final volume.",
                "reagents": ["PCR Master Mix", "forward primer", "reverse primer", "template DNA", "nuclease-free water"],
                "equipment": ["PCR tubes", "micropipettes", "ice bucket"],
                "timing": "5 minutes",
                "temperature": "on ice",
                "safety_level": "low",
                "tips": "Keep all reagents on ice. Mix gently by flicking, do not vortex. Always use filter tips to prevent contamination."
            },
            {
                "step_number": 2,
                "title": "Program Thermocycler",
                "description": "Set up the PCR thermocycler with the following program: Initial denaturation at 95°C for 5 minutes, then 30 cycles of: denaturation at 95°C for 30 seconds, annealing at 55-65°C for 30 seconds (adjust based on primer Tm), extension at 72°C for 1 minute per kb of product. Final extension at 72°C for 10 minutes, then hold at 4°C.",
                "reagents": [],
                "equipment": ["PCR thermocycler"],
                "timing": "2-3 hours total",
                "temperature": "95°C / 55-65°C / 72°C cycling",
                "safety_level": "low",
                "tips": "Calculate annealing temperature as 5°C below the lowest primer Tm. For GC-rich templates, use a higher denaturation temp or add DMSO."
            },
            {
                "step_number": 3,
                "title": "Run PCR",
                "description": "Place PCR tubes in the thermocycler and start the program. Ensure the lid is heated to 105°C to prevent condensation.",
                "reagents": [],
                "equipment": ["PCR thermocycler"],
                "timing": "2-3 hours",
                "temperature": "varies by program",
                "safety_level": "low",
                "tips": "Make sure tubes are properly sealed. Include a negative control (no template) and positive control."
            },
            {
                "step_number": 4,
                "title": "Analyze Results",
                "description": "After PCR is complete, analyze 5 µL of the product on a 1% agarose gel with ethidium bromide staining. Compare band size against DNA ladder.",
                "reagents": ["agarose", "TAE buffer", "ethidium bromide", "DNA ladder", "loading dye"],
                "equipment": ["gel electrophoresis apparatus", "UV transilluminator", "micropipettes"],
                "timing": "45 minutes",
                "temperature": "room temperature",
                "safety_level": "medium",
                "tips": "EtBr is a mutagen — wear gloves. Alternatively, use SYBR Safe for lower toxicity. Do not look directly at UV without protection."
            }
        ],
        "reagents": ["PCR Master Mix", "forward primer", "reverse primer", "template DNA", "nuclease-free water", "agarose", "TAE buffer", "ethidium bromide", "DNA ladder", "loading dye"],
        "equipment": ["PCR thermocycler", "micropipettes", "gel electrophoresis apparatus", "UV transilluminator", "ice bucket"],
    },
    {
        "title": "Agarose Gel Electrophoresis Protocol",
        "experiment_type": "gel_electrophoresis",
        "steps": [
            {
                "step_number": 1,
                "title": "Prepare Agarose Gel",
                "description": "Weigh agarose powder for desired concentration (0.8-2% depending on DNA size). For 1% gel: dissolve 1g agarose in 100mL 1x TAE buffer. Microwave for 1-2 minutes until fully dissolved, swirling occasionally. Let cool to ~55°C.",
                "reagents": ["agarose", "1x TAE buffer"],
                "equipment": ["microwave", "Erlenmeyer flask", "balance"],
                "timing": "10 minutes",
                "temperature": "55°C for pouring",
                "safety_level": "medium",
                "tips": "Hot agarose can cause burns. Use heat-resistant gloves. Do not let it boil over in the microwave."
            },
            {
                "step_number": 2,
                "title": "Cast Gel",
                "description": "Add ethidium bromide (0.5 µg/mL) or SYBR Safe to cooled agarose. Pour into gel tray with comb(s) in place. Let solidify for 20-30 minutes at room temperature.",
                "reagents": ["ethidium bromide or SYBR Safe"],
                "equipment": ["gel casting tray", "comb"],
                "timing": "30 minutes",
                "temperature": "room temperature",
                "safety_level": "high",
                "tips": "If using EtBr, wear gloves at all times. Pour in designated EtBr area. SYBR Safe is a safer alternative."
            },
            {
                "step_number": 3,
                "title": "Load Samples",
                "description": "Mix DNA samples with 6x loading dye (1:5 ratio). Load 5-20 µL per well. Load DNA ladder in the first and/or last lane. Submerge gel in 1x TAE buffer in the electrophoresis tank.",
                "reagents": ["6x loading dye", "DNA ladder"],
                "equipment": ["gel electrophoresis tank", "micropipettes"],
                "timing": "5 minutes",
                "temperature": "room temperature",
                "safety_level": "medium",
                "tips": "Load samples slowly to avoid overflowing wells. Use a steady hand and brace your pipetting arm."
            },
            {
                "step_number": 4,
                "title": "Run Electrophoresis",
                "description": "Connect electrodes (DNA runs toward the positive/red electrode). Run at 80-120V for 30-60 minutes. Check that bubbles are forming at electrodes.",
                "reagents": [],
                "equipment": ["power supply", "gel electrophoresis tank"],
                "timing": "30-60 minutes",
                "temperature": "room temperature",
                "safety_level": "medium",
                "tips": "Never touch the buffer or electrodes while the power supply is on. Higher voltage = faster but less resolution."
            },
            {
                "step_number": 5,
                "title": "Visualize Results",
                "description": "Place gel on UV transilluminator. Photograph the gel with a gel documentation system. Identify bands by comparison with DNA ladder.",
                "reagents": [],
                "equipment": ["UV transilluminator", "gel documentation system"],
                "timing": "5 minutes",
                "temperature": "room temperature",
                "safety_level": "medium",
                "tips": "Wear UV-protective goggles. Minimize UV exposure time to reduce DNA damage if you plan to cut bands."
            }
        ],
        "reagents": ["agarose", "1x TAE buffer", "ethidium bromide", "SYBR Safe", "6x loading dye", "DNA ladder"],
        "equipment": ["microwave", "gel casting tray", "gel electrophoresis tank", "power supply", "UV transilluminator"],
    },
    {
        "title": "Bacterial Transformation Protocol (Heat Shock)",
        "experiment_type": "bacterial_transformation",
        "steps": [
            {
                "step_number": 1,
                "title": "Thaw Competent Cells",
                "description": "Remove competent E. coli cells from -80°C freezer. Thaw on ice for 20-30 minutes. Do not warm with hands.",
                "reagents": ["competent E. coli cells"],
                "equipment": ["ice bucket", "-80°C freezer"],
                "timing": "20-30 minutes",
                "temperature": "on ice",
                "safety_level": "low",
                "tips": "Competent cells are very fragile. Never vortex. Keep on ice at all times. Handle gently."
            },
            {
                "step_number": 2,
                "title": "Add Plasmid DNA",
                "description": "Add 1-5 µL of plasmid DNA (1-100 ng) to 50 µL competent cells. Mix gently by flicking the tube. Do NOT pipette up and down.",
                "reagents": ["plasmid DNA"],
                "equipment": ["micropipettes", "1.5 mL microcentrifuge tubes"],
                "timing": "2 minutes",
                "temperature": "on ice",
                "safety_level": "low",
                "tips": "Use as little DNA as possible for highest efficiency. Include a no-DNA negative control."
            },
            {
                "step_number": 3,
                "title": "Incubate on Ice",
                "description": "Incubate the DNA-cell mixture on ice for 30 minutes.",
                "reagents": [],
                "equipment": ["ice bucket", "timer"],
                "timing": "30 minutes",
                "temperature": "0°C (ice)",
                "safety_level": "low",
                "tips": "Do not disturb the tubes during incubation. This step allows DNA to adsorb to the cell surface."
            },
            {
                "step_number": 4,
                "title": "Heat Shock",
                "description": "Transfer tubes to a 42°C water bath for exactly 45 seconds. Immediately return to ice for 2 minutes.",
                "reagents": [],
                "equipment": ["water bath (42°C)", "timer"],
                "timing": "45 seconds + 2 minutes",
                "temperature": "42°C then ice",
                "safety_level": "low",
                "tips": "Timing is critical! Too long at 42°C will kill cells. Use a timer. The temperature shock creates pores in the cell membrane."
            },
            {
                "step_number": 5,
                "title": "Recovery",
                "description": "Add 950 µL warm SOC or LB medium (37°C) to each tube. Incubate at 37°C with shaking (225 rpm) for 1 hour.",
                "reagents": ["SOC medium or LB medium"],
                "equipment": ["shaking incubator (37°C)", "micropipettes"],
                "timing": "1 hour",
                "temperature": "37°C",
                "safety_level": "low",
                "tips": "Recovery step allows cells to express the antibiotic resistance gene before plating on selective media."
            },
            {
                "step_number": 6,
                "title": "Plate Cells",
                "description": "Spread 50-200 µL of transformed cells on pre-warmed LB agar plates containing the appropriate antibiotic. Incubate plates inverted at 37°C overnight (12-16 hours).",
                "reagents": ["LB agar plates with antibiotic"],
                "equipment": ["cell spreader or glass beads", "incubator (37°C)"],
                "timing": "overnight (12-16 hours)",
                "temperature": "37°C",
                "safety_level": "low",
                "tips": "Invert plates to prevent condensation from dripping onto colonies. Use proper sterile technique."
            }
        ],
        "reagents": ["competent E. coli cells", "plasmid DNA", "SOC medium", "LB agar plates", "antibiotic"],
        "equipment": ["ice bucket", "-80°C freezer", "water bath", "shaking incubator", "micropipettes"],
    },
    {
        "title": "Genomic DNA Extraction Protocol (Bacterial)",
        "experiment_type": "dna_extraction",
        "steps": [
            {
                "step_number": 1,
                "title": "Harvest Cells",
                "description": "Pellet 1-5 mL of overnight bacterial culture by centrifugation at 6,000 x g for 5 minutes. Discard supernatant.",
                "reagents": ["overnight bacterial culture"],
                "equipment": ["centrifuge", "microcentrifuge tubes"],
                "timing": "5 minutes",
                "temperature": "room temperature",
                "safety_level": "low",
                "tips": "Make sure the pellet is visible. If culture is dilute, use more volume."
            },
            {
                "step_number": 2,
                "title": "Lyse Cells",
                "description": "Resuspend pellet in 200 µL lysis buffer (10 mM Tris-HCl pH 8.0, 1 mM EDTA, 0.5% SDS). Add 20 µL Proteinase K (20 mg/mL). Incubate at 56°C for 1 hour.",
                "reagents": ["lysis buffer", "Proteinase K"],
                "equipment": ["heat block or water bath (56°C)", "micropipettes"],
                "timing": "1 hour",
                "temperature": "56°C",
                "safety_level": "medium",
                "tips": "SDS can irritate skin. Wear gloves. Mix gently — do not vortex to avoid shearing DNA."
            },
            {
                "step_number": 3,
                "title": "Remove RNA",
                "description": "Add 5 µL RNase A (10 mg/mL). Incubate at 37°C for 30 minutes.",
                "reagents": ["RNase A"],
                "equipment": ["heat block or incubator (37°C)"],
                "timing": "30 minutes",
                "temperature": "37°C",
                "safety_level": "low",
                "tips": "This step removes RNA contamination which can interfere with downstream applications."
            },
            {
                "step_number": 4,
                "title": "Purify DNA",
                "description": "Add equal volume of phenol:chloroform:isoamyl alcohol (25:24:1). Vortex for 15 seconds. Centrifuge at 12,000 x g for 10 minutes. Transfer the upper aqueous phase to a new tube.",
                "reagents": ["phenol:chloroform:isoamyl alcohol"],
                "equipment": ["centrifuge", "microcentrifuge tubes", "fume hood"],
                "timing": "15 minutes",
                "temperature": "room temperature",
                "safety_level": "critical",
                "tips": "PHENOL IS HIGHLY TOXIC AND CORROSIVE. Work in fume hood. Wear gloves, goggles, and lab coat. Avoid skin contact."
            },
            {
                "step_number": 5,
                "title": "Precipitate DNA",
                "description": "Add 0.1 volume 3M sodium acetate (pH 5.2) and 2.5 volumes cold 100% ethanol. Mix gently. Incubate at -20°C for 1 hour or overnight.",
                "reagents": ["3M sodium acetate", "100% ethanol"],
                "equipment": ["-20°C freezer"],
                "timing": "1 hour to overnight",
                "temperature": "-20°C",
                "safety_level": "medium",
                "tips": "Ethanol is flammable. Keep away from open flames. Cold ethanol works better for precipitation."
            },
            {
                "step_number": 6,
                "title": "Wash and Dissolve",
                "description": "Centrifuge at 12,000 x g for 15 minutes at 4°C. Discard supernatant. Wash pellet with 500 µL cold 70% ethanol. Air-dry pellet for 5-10 minutes. Dissolve in 50-100 µL TE buffer or nuclease-free water.",
                "reagents": ["70% ethanol", "TE buffer"],
                "equipment": ["centrifuge (refrigerated)", "micropipettes"],
                "timing": "30 minutes",
                "temperature": "4°C centrifugation, RT for drying",
                "safety_level": "low",
                "tips": "Do not over-dry the pellet — it becomes difficult to dissolve. The pellet may be invisible but is present."
            }
        ],
        "reagents": ["lysis buffer", "Proteinase K", "RNase A", "phenol:chloroform:isoamyl alcohol", "sodium acetate", "ethanol", "TE buffer"],
        "equipment": ["centrifuge", "heat block", "fume hood", "-20°C freezer", "micropipettes"],
    },
    {
        "title": "Basic Cell Culture — Subculturing Adherent Cells",
        "experiment_type": "cell_culture",
        "steps": [
            {
                "step_number": 1,
                "title": "Prepare Workspace",
                "description": "Disinfect the biosafety cabinet (BSC) with 70% ethanol. Turn on UV for 15 minutes before use. Warm complete growth medium and trypsin to 37°C in a water bath.",
                "reagents": ["70% ethanol", "complete growth medium", "trypsin-EDTA"],
                "equipment": ["biosafety cabinet", "water bath (37°C)"],
                "timing": "20 minutes",
                "temperature": "37°C for reagent warming",
                "safety_level": "medium",
                "tips": "Always work in the BSC for cell culture. UV exposure: never work under UV. Spray everything with 70% ethanol before placing in BSC."
            },
            {
                "step_number": 2,
                "title": "Remove Old Media",
                "description": "Aspirate spent culture media from the flask. Be careful not to touch the cell layer with the aspirator tip.",
                "reagents": [],
                "equipment": ["vacuum aspirator", "aspirator tips"],
                "timing": "1 minute",
                "temperature": "room temperature",
                "safety_level": "low",
                "tips": "Tilt the flask and aspirate from the corner opposite the cells."
            },
            {
                "step_number": 3,
                "title": "Wash Cells",
                "description": "Gently wash cells with 5-10 mL PBS (without calcium/magnesium) to remove residual serum that inhibits trypsin.",
                "reagents": ["PBS (Ca/Mg-free)"],
                "equipment": ["serological pipettes"],
                "timing": "1 minute",
                "temperature": "room temperature",
                "safety_level": "low",
                "tips": "Add PBS to the side of the flask, not directly onto cells."
            },
            {
                "step_number": 4,
                "title": "Trypsinize Cells",
                "description": "Add 2-3 mL warm trypsin-EDTA (0.25%). Incubate at 37°C for 3-5 minutes. Check detachment under microscope. Tap flask gently to dislodge remaining cells.",
                "reagents": ["trypsin-EDTA (0.25%)"],
                "equipment": ["incubator (37°C)", "inverted microscope"],
                "timing": "3-5 minutes",
                "temperature": "37°C",
                "safety_level": "low",
                "tips": "Do not over-trypsinize — it damages cells. Check every 2 minutes. Rounded cells = detaching."
            },
            {
                "step_number": 5,
                "title": "Neutralize and Collect",
                "description": "Add equal volume of complete medium (with serum) to neutralize trypsin. Pipette up and down to create single-cell suspension. Transfer to a centrifuge tube.",
                "reagents": ["complete growth medium"],
                "equipment": ["centrifuge tubes", "serological pipettes"],
                "timing": "2 minutes",
                "temperature": "room temperature",
                "safety_level": "low",
                "tips": "Serum in the medium neutralizes trypsin. Pipette gently to avoid cell damage."
            },
            {
                "step_number": 6,
                "title": "Seed New Flask",
                "description": "Count cells if needed. Seed new flask at appropriate density (typically 1:3 to 1:10 split ratio). Add fresh complete medium to appropriate volume. Label flask with cell line, passage number, date, and initials.",
                "reagents": ["complete growth medium"],
                "equipment": ["new culture flask", "hemocytometer (optional)"],
                "timing": "5 minutes",
                "temperature": "37°C incubator after seeding",
                "safety_level": "low",
                "tips": "Passage number should be recorded. Most cell lines should not exceed passage 30. Check cells next day for attachment."
            }
        ],
        "reagents": ["70% ethanol", "complete growth medium", "trypsin-EDTA", "PBS"],
        "equipment": ["biosafety cabinet", "CO2 incubator", "water bath", "inverted microscope", "vacuum aspirator"],
    },
]


async def seed_protocols():
    """Seed the protocol database with common synthetic biology protocols."""
    from pinecone import Pinecone

    settings = get_settings()

    logger.info("Connecting to Pinecone...")
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)

    for protocol in SEED_PROTOCOLS:
        logger.info("Seeding protocol: %s", protocol["title"])

        protocol_id = str(uuid.uuid4())
        vectors_to_upsert = []

        for step in protocol["steps"]:
            # Create text representation for embedding
            step_text = f"Protocol: {protocol['title']}\n"
            step_text += f"Step {step['step_number']}: {step['title']}\n"
            step_text += f"Description: {step['description']}\n"
            if step.get("tips"):
                step_text += f"Tips: {step['tips']}\n"
            if step.get("reagents"):
                step_text += f"Reagents: {', '.join(step['reagents'])}\n"

            # Generate embedding
            embedding = embed_texts([step_text])[0]

            chunk_id = f"{protocol_id}_step_{step['step_number']}"

            vectors_to_upsert.append({
                "id": chunk_id,
                "values": embedding,
                "metadata": {
                    "text": step_text,
                    "protocol_id": protocol_id,
                    "protocol_title": protocol["title"],
                    "chunk_type": "step",
                    "experiment_type": protocol["experiment_type"],
                    "step_number": step["step_number"],
                    "step_title": step["title"],
                    "reagents": step.get("reagents", []),
                    "equipment": step.get("equipment", []),
                    "timing": step.get("timing", ""),
                    "temperature": step.get("temperature", ""),
                    "safety_level": step.get("safety_level", "low"),
                    "tips": step.get("tips", ""),
                    "source": "seed_library",
                },
            })

        # Also add an overview chunk
        overview_text = f"Protocol: {protocol['title']}\n"
        overview_text += f"Experiment Type: {protocol['experiment_type']}\n"
        overview_text += f"Total Steps: {len(protocol['steps'])}\n"
        overview_text += f"Required Reagents: {', '.join(protocol.get('reagents', []))}\n"
        overview_text += f"Required Equipment: {', '.join(protocol.get('equipment', []))}\n"

        overview_embedding = embed_texts([overview_text])[0]
        vectors_to_upsert.append({
            "id": f"{protocol_id}_overview",
            "values": overview_embedding,
            "metadata": {
                "text": overview_text,
                "protocol_id": protocol_id,
                "protocol_title": protocol["title"],
                "chunk_type": "overview",
                "experiment_type": protocol["experiment_type"],
                "reagents": protocol.get("reagents", []),
                "equipment": protocol.get("equipment", []),
                "source": "seed_library",
            },
        })

        # Upsert to Pinecone in batches
        namespace = "protocols"
        for i in range(0, len(vectors_to_upsert), PINECONE_BATCH_SIZE):
            batch = vectors_to_upsert[i : i + PINECONE_BATCH_SIZE]
            index.upsert(vectors=batch, namespace=namespace)

        logger.info(
            "  ✅ Seeded %d chunks for '%s'",
            len(vectors_to_upsert),
            protocol["title"],
        )

    # Verify
    stats = index.describe_index_stats()
    logger.info("\n📊 Final index stats: %s", stats)
    logger.info("✅ All protocols seeded successfully!")


if __name__ == "__main__":
    asyncio.run(seed_protocols())
