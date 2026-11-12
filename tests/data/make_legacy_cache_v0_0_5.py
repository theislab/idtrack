"""Write ``legacy_cache_v0.0.5.h5`` with the table writer of idtrack 0.0.5.

Run it with a checkout of the ``v0.0.5`` tag and the environment of that time; the shipped file was written under
pandas 2.2.3, numpy 1.26.4 and h5py 3.13.0 with HDF5 1.14.6::

    git archive v0.0.5 idtrack | tar -x -C <dir>
    python -I tests/data/make_legacy_cache_v0_0_5.py <dir> tests/data/legacy_cache_v0.0.5.h5

The table is shaped like a cached raw Ensembl table: stable ids with missing values (``stable_id_event``), versions
with NaN, integer keys, an empty string, a categorical and a boolean.
"""

import sys

sys.path.insert(0, sys.argv[1])

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from idtrack import _utils_hdf5 as hs  # noqa: E402

df = pd.DataFrame(
    {
        "old_stable_id": pd.Series(["ENSSSCG00000000001", None, "ENSSSCG00000000003", np.nan], dtype=object),
        "old_version": [1.0, np.nan, 2.0, 1.0],
        "new_stable_id": pd.Series(
            ["ENSSSCG00000000001", "ENSSSCG00000000002", "", "ENSSSCG00000000004"], dtype=object
        ),
        "mapping_session_id": np.array([10, 11, 12, 13], dtype=np.int64),
        "type": pd.Categorical(["gene", "gene", "transcript", "gene"]),
        "is_current": [True, False, True, True],
    }
)
hs.to_hdf(sys.argv[2], "ens115_mysql_stable_id_event", df, mode="w")
print(pd.__version__, np.__version__, h5py.__version__, h5py.version.hdf5_version)
