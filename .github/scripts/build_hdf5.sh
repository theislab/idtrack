#!/usr/bin/env bash
# Build the HDF5 library from the HDF Group's release source into a prefix, unless the prefix holds it already.
#
# Usage: build_hdf5.sh <version> <prefix>      (e.g. build_hdf5.sh 2.0.0 "$HOME/hdf5-2.0.0")
#
# Used by the CI jobs of Python versions that h5py publishes no wheels for: pip then builds h5py from source, and it
# must link the same HDF5 that the h5py wheels of the other Python versions bundle, not the runner's distribution
# HDF5 (1.10 on ubuntu-22.04, which cannot read files that HDF5 1.12 and later write with the newest format). The
# steps follow h5py's own CI (ci/get_hdf5_if_needed.sh). When GITHUB_ENV is set, the variables that make pip build
# h5py against this HDF5 are exported to the following steps.
set -euo pipefail

version="$1"
prefix="$2"

if [ ! -f "$prefix/lib/libhdf5.so" ]; then
    work="$(mktemp -d)"
    # The release tags are named `2.0.0` (2.x), `hdf5_1.14.6` or `hdf5-1_10_7` (older releases).
    IFS=. read -r major minor patch <<< "$version"
    downloaded=""
    for tag in "$version" "hdf5_${version}" "hdf5-${major}_${minor}_${patch}"; do
        if curl --location --fail --silent --show-error \
            "https://github.com/HDFGroup/hdf5/archive/refs/tags/${tag}.tar.gz" --output "$work/hdf5.tar.gz"; then
            downloaded="$tag"
            break
        fi
    done
    [ -n "$downloaded" ] || { echo "no HDF5 release source found for version $version" >&2; exit 1; }
    mkdir -p "$work/src"
    tar -xzf "$work/hdf5.tar.gz" --strip-components=1 -C "$work/src"
    cmake -S "$work/src" -B "$work/build" \
        -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_INSTALL_PREFIX="$prefix" \
        -D BUILD_TESTING=OFF \
        -D BUILD_STATIC_LIBS=OFF \
        -D HDF5_BUILD_EXAMPLES=OFF \
        -D HDF5_BUILD_TOOLS=OFF \
        -D HDF5_BUILD_UTILS=OFF \
        -D HDF5_ENABLE_ZLIB_SUPPORT=ON \
        -D HDF5_ENABLE_Z_LIB_SUPPORT=ON  # the name of the same option before HDF5 2.0 (CMake ignores the other one)
    make -C "$work/build" -j "$(nproc)"
    make -C "$work/build" install
    rm -rf "$work"
fi

if [ -n "${GITHUB_ENV:-}" ]; then
    {
        echo "HDF5_DIR=$prefix"
        echo "HDF5_VERSION=$version"
        echo "LD_LIBRARY_PATH=$prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        # A wheel of h5py that pip built earlier against another HDF5 must not be reused from the pip cache.
        echo "PIP_NO_CACHE_DIR=1"
    } >> "$GITHUB_ENV"
fi
