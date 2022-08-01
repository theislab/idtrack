#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

import os

import idtrack

organism = "human"
id_form = "gene"
local_dir = os.path.join(os.path.dirname(__file__), "temp")
dm, vdf, st, g = idtrack.initialize_minimal(
    organism,
    id_form,
    local_dir,
    ignore_before=105,
    ignore_after=106,
    initialize_datasets=True,
    initialize_graph=True,
    clean_up=True,
)
