"""
STEP 2: Standardize classes

DeepFish classes:   {0: "fish"}
Fish4Knowledge:     {0,1,2,5,6,7,8} -> mapped to fish
TrashCan classes:   full list below

Final unified classes (marine detection):
  0  fish
  1  trash_bag
  2  trash_bottle
  3  trash_can
  4  trash_cup
  5  trash_net
  6  trash_rope
  7  trash_pipe
  8  trash_wreckage
  9  trash_other
  10 crab
  11 starfish
  12 eel
  13 echinus        (sea urchin type)
  14 holothurian    (sea cucumber)
  15 scallop
  16 animal_other
"""

# ─── FINAL CLASS LIST ───────────────────────────────────────────────────────
FINAL_CLASSES = [
    "fish",           # 0
    "trash_bag",      # 1
    "trash_bottle",   # 2
    "trash_can",      # 3
    "trash_cup",      # 4
    "trash_net",      # 5
    "trash_rope",     # 6
    "trash_pipe",     # 7
    "trash_wreckage", # 8
    "trash_other",    # 9
    "crab",           # 10
    "starfish",       # 11
    "eel",            # 12
    "echinus",        # 13
    "holothurian",    # 14
    "scallop",        # 15
    "animal_other",   # 16
]

# ─── DEEPFISH MAPPING ───────────────────────────────────────────────────────
# Original: {0: unnamed fish class}
DEEPFISH_MAP = {
    0: 0,   # fish -> fish
}

# ─── FISH4KNOWLEDGE MAPPING ─────────────────────────────────────────────────
# Original classes 0,1,2,5,6,7,8 are all fish species
FISH4KNOWLEDGE_MAP = {
    0: 0,   # fish
    1: 0,   # fish
    2: 0,   # fish
    5: 0,   # fish
    6: 0,   # fish
    7: 0,   # fish
    8: 0,   # fish
}

# ─── TRASHCAN MAPPING ───────────────────────────────────────────────────────
# Original TrashCan class index -> name (from data.yaml order)
TRASHCAN_ORIGINAL = [
    "plant",                  # 0  -> DROP (not a detection target)
    "starfish",               # 1  -> starfish (11)
    "animal_crab",            # 2  -> crab (10)
    "animal_eel",             # 3  -> eel (12)
    "animal_etc",             # 4  -> animal_other (16)
    "animal_fish",            # 5  -> fish (0)
    "animal_shells",          # 6  -> animal_other (16)
    "animal_starfish",        # 7  -> starfish (11)
    "echinus",                # 8  -> echinus (13)
    "holothurian",            # 9  -> holothurian (14)
    "rov",                    # 10 -> DROP (ROV equipment, not target)
    "scallop",                # 11 -> scallop (15)
    "seacucumber",            # 12 -> holothurian (14)
    "seaurchin",              # 13 -> echinus (13)
    "trash_bag",              # 14 -> trash_bag (1)
    "trash_bottle",           # 15 -> trash_bottle (2)
    "trash_branch",           # 16 -> trash_other (9)
    "trash_can",              # 17 -> trash_can (3)
    "trash_clothing",         # 18 -> trash_other (9)
    "trash_container",        # 19 -> trash_other (9)
    "trash_cup",              # 20 -> trash_cup (4)
    "trash_net",              # 21 -> trash_net (5)
    "trash_pipe",             # 22 -> trash_pipe (7)
    "trash_rope",             # 23 -> trash_rope (6)
    "trash_snack_wrapper",    # 24 -> trash_other (9)
    "trash_tarp",             # 25 -> trash_other (9)
    "trash_unknown_instance", # 26 -> trash_other (9)
    "trash_wreckage",         # 27 -> trash_wreckage (8)
]

TRASHCAN_MAP = {
    0:  None,  # plant -> DROP
    1:  11,    # starfish
    2:  10,    # crab
    3:  12,    # eel
    4:  16,    # animal_other
    5:  0,     # fish
    6:  16,    # animal_other
    7:  11,    # starfish
    8:  13,    # echinus
    9:  14,    # holothurian
    10: None,  # rov -> DROP
    11: 15,    # scallop
    12: 14,    # holothurian (seacucumber)
    13: 13,    # echinus (seaurchin)
    14: 1,     # trash_bag
    15: 2,     # trash_bottle
    16: 9,     # trash_other (branch)
    17: 3,     # trash_can
    18: 9,     # trash_other (clothing)
    19: 9,     # trash_other (container)
    20: 4,     # trash_cup
    21: 5,     # trash_net
    22: 7,     # trash_pipe
    23: 6,     # trash_rope
    24: 9,     # trash_other (snack wrapper)
    25: 9,     # trash_other (tarp)
    26: 9,     # trash_other (unknown)
    27: 8,     # trash_wreckage
}

# ─── DATASET CONFIG ──────────────────────────────────────────────────────────
DATASET_CONFIG = {
    "DeepFish":       {"root": "data/DeepFish",       "class_map": DEEPFISH_MAP},
    "Fish4Knowledge": {"root": "data/Fish4Knowledge", "class_map": FISH4KNOWLEDGE_MAP},
    "TrashCan":       {"root": "data/TrashCan",       "class_map": TRASHCAN_MAP},
}

if __name__ == "__main__":
    print("Final class list:")
    for i, cls in enumerate(FINAL_CLASSES):
        print(f"  {i:2d}: {cls}")
    print(f"\nTotal classes: {len(FINAL_CLASSES)}")
    print("\nTrashCan dropped classes:")
    for orig_id, new_id in TRASHCAN_MAP.items():
        if new_id is None:
            print(f"  [{orig_id}] {TRASHCAN_ORIGINAL[orig_id]} -> DROPPED")
    print("\n[DONE] Step 2 config ready. Import DATASET_CONFIG in step 3.")
