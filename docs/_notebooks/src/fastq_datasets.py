#!/usr/bin/env python3

from collections.abc import Mapping
from typing import Dict

DatasetSpec = Mapping[str, object]

fastq_datasets: dict[str, DatasetSpec] = {
    "pbmc_1k_v3": {
        "title": "1k PBMCs from a Healthy Donor (v3 chemistry)",
        "organism": "homo_sapiens",
        "assay": "3p",
        # Was: "3' v3" -> must be Cell Ranger token:
        # 3' v3  ==> SC3Pv3  (Next GEM kit)
        "chemistry": "SC3Pv3",
        "runner": "cellranger",
        "mode": "count",
        "expected_cells": 1222,
        "mean_reads_per_cell": 54502,
        "params": {
            "expect-cells": "1000",
            "create-bam": "false",
            # add more when needed (e.g., "transcriptome": "/path/to/ref")
        },
        "fastqs": {
            "url": "https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_1k_v3/pbmc_1k_v3_fastqs.tar",
            "md5": "265ebe8f77ad90db350984d9c7a59e52",
            "filename": "pbmc_1k_v3_fastqs.tar",
            "archive": "tar",
            "size_gb": 5.5,
        },
        "aux": {
            # no extras for this dataset
        },
        "page_url": "https://www.10xgenomics.com/datasets/1-k-pbm-cs-from-a-healthy-donor-v-3-chemistry-3-standard-3-0-0",
        "notes": "",
    },
    "pbmc_20k_donors1_4_multiplex_gemx_3p": {
        "title": "20k Human PBMCs Multiplex Sample (Donors 1–4, GEM-X 3')",
        "organism": "homo_sapiens",
        "assay": "3p",
        # GEM‑X 3' corresponds to 3' v4 => SC3Pv4
        "chemistry": "SC3Pv4",
        "runner": "cellranger",
        "mode": "multi",
        "expected_cells": 20166,
        "mean_reads_per_cell": 53529,
        "params": {
            "create-bam": "false"
            # For 'multi', params typically come from the CSV; keep CLI params minimal here.
        },
        "fastqs": {
            "url": "https://s3-us-west-2.amazonaws.com/10x.files/samples/cell-exp/9.0.0/20k_Human_Donor1-4_PBMC_3p_gem-x_multiplex_Multiplex/20k_Human_Donor1-4_PBMC_3p_gem-x_multiplex_Multiplex_fastqs.tar",
            "md5": "95d657fc40c077a992e75ac36ce9d1f7",
            "filename": "20k_Human_Donor1-4_PBMC_3p_gem-x_multiplex_Multiplex_fastqs.tar",
            "archive": "tar",
            "size_gb": 86.9,
        },
        "aux": {
            "multi_csv": {
                "url": "https://cf.10xgenomics.com/samples/cell-exp/9.0.0/20k_Human_Donor1-4_PBMC_3p_gem-x_multiplex_Multiplex/20k_Human_Donor1-4_PBMC_3p_gem-x_multiplex_Multiplex_config.csv",
                "md5": "1aab509534290736a86030ca184b0055",
                "filename": "20k_Human_Donor1-4_PBMC_3p_gem-x_multiplex_Multiplex_config.csv",
                "archive": "csv",
            }
        },
        "page_url": "https://www.10xgenomics.com/datasets/20k_Human_Donor1-4_PBMC_3p_gem-x_multiplex",
        "notes": "",
    },
    "pbmc_10k_5p_v3_ultima": {
        "title": "10k Human PBMCs, Ultima Sequencing",
        "organism": "homo_sapiens",
        "assay": "5p",
        # 5' v3 GEM‑X on Ultima uses R2‑only alignment => SC5P-R2-v3
        "chemistry": "SC5P-R2-v3",
        "expected_cells": 9506,
        "mean_reads_per_cell": 42079,
        "runner": "cellranger",
        "mode": "multi",  # per dataset page (Universal 5' uses multi)
        "params": {"create-bam": "false"},  # provided via multi CSV when available
        "fastqs": {
            "url": "https://s3-us-west-2.amazonaws.com/10x.files/samples/cell-vdj/9.0.1/10k_Human_PBMC_5p_v3_Ultima_Multiplex/10k_Human_PBMC_5p_v3_Ultima_Multiplex_fastqs.tar",
            "md5": "ffde7b5ce12698dedc06580ee73d04b8",
            "filename": "10k_Human_PBMC_5p_v3_Ultima_Multiplex_fastqs.tar",
            "archive": "tar",
            "size_gb": 35.1,
        },
        "aux": {
            "multi_csv": {
                "url": "https://cf.10xgenomics.com/samples/cell-vdj/9.0.1/10k_Human_PBMC_5p_v3_Ultima_Multiplex/10k_Human_PBMC_5p_v3_Ultima_Multiplex_config.csv",
                "md5": "2b04b6e8a6cf7d72475ab394cb0c671d",
                "filename": "10k_Human_PBMC_5p_v3_Ultima_Multiplex_config.csv",
                "archive": "csv",
            }
        },
        "page_url": "https://www.10xgenomics.com/datasets/10k_Human_PBMC_5p_v3_Ultima",
        "notes": "",
    },
    "pbmc_10k_3p_v31_si": {
        "title": "10k PBMCs (3' v3.1, Single Index)",
        "organism": "homo_sapiens",
        "assay": "3p",
        # 3' v3.1 (Next GEM SI) is covered by SC3Pv3 in CLI tokens
        "chemistry": "SC3Pv3",
        "runner": "cellranger",
        "mode": "count",
        "expected_cells": 10985,
        "mean_reads_per_cell": 171996,
        "params": {"expect-cells": "10000", "create-bam": "false"},
        "fastqs": {
            "url": "https://s3-us-west-2.amazonaws.com/10x.files/samples/cell-exp/4.0.0/SC3_v3_NextGem_SI_PBMC_10K/SC3_v3_NextGem_SI_PBMC_10K_fastqs.tar",
            "md5": "38d2d253f8537d3c39aaa5832da39ad3",
            "filename": "SC3_v3_NextGem_SI_PBMC_10K_fastqs.tar",
            "archive": "tar",
            "size_gb": 145.0,
        },
        "aux": {},
        "page_url": "https://www.10xgenomics.com/datasets/10-k-peripheral-blood-mononuclear-cells-pbm-cs-from-a-healthy-donor-single-indexed-3-1-standard-4-0-0",
        "notes": "",
    },
}
