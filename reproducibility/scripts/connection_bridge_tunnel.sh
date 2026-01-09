#!/usr/bin/env bash

set -euo pipefail

################################################################################
# connection_bridge_tunnel.sh
#
# Beginner-friendly helper for `idtrack.ConnectionBridge`.
#
# This script does NOT start Jupyter.
# It prints (to stdout) the ONE `ssh ...` command you need to run from your LOCAL
# machine to create a SOCKS5 proxy on the REMOTE machine where your Python code
# actually runs (server/login node or compute node).
#
# ------------------------------------------------------------------------------
# Mental model (why this feels “weird” at first)
# ------------------------------------------------------------------------------
#
# There are two different directions of network traffic involved when you use a
# cluster:
#
#   1) INBOUND (browser -> Jupyter)
#      Your laptop browser needs to reach a Jupyter server running remotely.
#      That is solved with an SSH *local forward* (ssh -L).
#
#   2) OUTBOUND (Python -> internet)
#      Your remote Python process needs to reach the public internet (Ensembl).
#      That is solved with an SSH *remote dynamic forward* (ssh -R PORT) which
#      creates a SOCKS proxy on the remote machine, plus `idtrack.ConnectionBridge`
#      inside Python which tells Python to use that proxy.
#
# These are independent problems, so it’s normal to have TWO ssh processes/tabs:
#   - one ssh process keeps the Jupyter browser tunnel open (-L)
#   - one ssh process keeps the SOCKS proxy tunnel open (-R)
#
# If (and only if) you can SSH directly to the machine running the kernel (often
# via `ssh -J login compute`), you can combine BOTH forwards (-L and -R) into a
# single ssh command. This script can generate that combined command too.
#
# ------------------------------------------------------------------------------
# What `ssh -R 127.0.0.1:1080 ...` actually does
# ------------------------------------------------------------------------------
#
# When you run (from your laptop):
#
#   ssh -N -R 127.0.0.1:1080 user@REMOTE
#
# SSH creates a SOCKS5 proxy listener *on REMOTE* at:
#
#   REMOTE:127.0.0.1:1080
#
# Any program running on REMOTE that connects to 127.0.0.1:1080 will have its
# network traffic forwarded through the SSH connection back to your laptop, and
# then out to the internet from your laptop.
#
# `idtrack.ConnectionBridge` is simply the “Python-side switch” that makes your
# Python process use that proxy (it monkeypatches `socket.socket` via PySocks).
#
# ------------------------------------------------------------------------------
# Three common scenarios
# ------------------------------------------------------------------------------
#
# A) Python runs on the server/login node (NO compute node):
#    - You create the proxy on the login node
#    - You run your Python script on the login node
#
# B) Python runs on a compute node (Slurm), NO Jupyter:
#    - You must create the proxy on the compute node (not just the login node!)
#    - You run your Python script on that compute node (e.g. sbatch/srun)
#
# C) Jupyter kernel runs on a compute node (browser on laptop):
#    - You still need the proxy on the compute node (because the kernel is there)
#    - You ALSO need a Jupyter browser tunnel
#
# In scenario C you can either:
#    - keep using your existing Jupyter tunnel script for INBOUND (-L), and open
#      a 2nd ssh tab for OUTBOUND (-R), OR
#    - use `--with-jupyter` to generate ONE combined ssh command (if your cluster
#      allows `ssh -J login compute`).
################################################################################

usage() {
  local script_name
  script_name="$(basename "$0")"
  cat <<EOF
Generate the SSH command needed by idtrack.ConnectionBridge (server/compute outbound internet).

This script runs on your LOCAL machine. It prints ONE ssh command (one line) that you run in another terminal tab.
Keep that ssh session open while your remote Python process runs.

What you get:
  - A SOCKS5 proxy on the REMOTE machine at 127.0.0.1:<socks_port>
  - Your Python process can then do:

      import idtrack
      with idtrack.ConnectionBridge(proxy_port=<socks_port>):
          ... run IDTrack ...

Usage:
  bash path/to/$script_name [options] <target> [target_user] [socks_port]

  # Or, if you want to run it directly:
  #   chmod +x path/to/$script_name
  #   path/to/$script_name [options] <target> [target_user] [socks_port]

Where:
  <target> can be:
    - a host name (e.g. compute123)
    - an SSH destination (e.g. user@compute123)
    - a Jupyter URL (e.g. "http://compute123:8888/lab?token=...") to auto-detect the compute host

Examples:
  # A) Python runs on a server/login node (no compute node)
  $script_name user@server

  # B) Python runs on a compute node (no Jupyter)
  $script_name --jump user@login compute123
  $script_name --jump user@login compute123 myclusteruser 1080

  # C) Jupyter kernel runs on compute node (browser on laptop)
  # (1) OUTBOUND proxy (this script) + (2) INBOUND Jupyter tunnel (your normal workflow)
  $script_name --jump user@login "http://compute123:8888/lab?token=..."

  # Optional: generate ONE combined ssh command with both -L (Jupyter) and -R (SOCKS)
  # This only works if you can SSH to the kernel host (often via ProxyJump).
  $script_name --jump user@login --with-jupyter "http://compute123:8888/lab?token=..."
  $script_name --jump user@login --with-jupyter --jupyter-local-port 9999 "http://compute123:8888/lab?token=..."

Step-by-step (beginner)
-----------------------
Scenario A — Python runs on the server/login node
  Local machine:
    1) Print the ssh command:
         bash path/to/$script_name user@server
    2) Paste the printed ssh command into a SECOND terminal tab and keep it open.
  Server:
    3) SSH in normally (your usual way).
    4) Run your Python script, enabling ConnectionBridge inside the process.

Scenario B — Python runs on a compute node (Slurm), not Jupyter
  1) Start a job on the cluster (interactive is easiest when learning):
       - Example ideas (cluster-specific): salloc / srun --pty bash / sbatch
  2) Find the compute hostname (examples): run `hostname` in the job, or check your job output.
  3) From your local machine, create the SOCKS proxy on THAT compute node:
       bash path/to/$script_name --jump user@login compute123
     Then paste/run the printed ssh command in a second tab and keep it open.
  4) Run your Python script on the compute node and enable ConnectionBridge at the start.

Scenario C — Jupyter kernel runs on a compute node (browser on your local machine)
  1) Start Jupyter on the cluster and copy the URL it prints (it contains the compute host and port).
  2) INBOUND: create your normal Jupyter browser tunnel (ssh -L ...) so your browser can open Jupyter.
  3) OUTBOUND: create the SOCKS proxy on the compute node using this script:
       bash path/to/$script_name --jump user@login "http://compute123:8888/lab?token=..."
     Then paste/run the printed ssh command in a second tab and keep it open.
  4) In the notebook kernel, enable ConnectionBridge so the kernel uses the proxy.

Options:
  -J, --jump <jump_host>        Jump host / login node (ProxyJump). Example: user@login
  -u, --user <target_user>      Username for the remote machine (if <target> does not include user@)
  -p, --socks-port <port>       SOCKS proxy port on the remote machine (default: 1080)
      --with-jupyter            Also include a Jupyter browser tunnel (-L) in the printed ssh command
      --jupyter-remote-port <p> Remote Jupyter port (defaults to the port parsed from the Jupyter URL)
      --jupyter-local-port <p>  Local port for your browser (defaults to same as remote port)
  -h, --help                    Show this help

Beginner checklist (when it fails):
  - The ssh session must be started from your LOCAL machine (the one with internet).
  - The SOCKS proxy must exist on the SAME machine as the Python process.
    If Python runs on a compute node, the proxy must be on that compute node.
  - In Python, enable ConnectionBridge before the first network access.
  - If you see: "ConnectionBridge requires PySocks", install it: `pip install PySocks`.

Notes:
  - On success this script prints ONLY the SSH command (one line) to stdout for easy copy/paste.
  - Warnings/errors go to stderr so they don't pollute the copy/paste line.
  - If your Jupyter URL uses host "localhost" or "127.0.0.1", the script cannot infer the compute hostname.
    In that case, pass the compute hostname directly (e.g. "compute123").
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

warn() {
  echo "Warning: $*" >&2
}

looks_like_url() {
  [[ "${1:-}" =~ ^https?:// ]]
}

is_integer() {
  [[ "${1:-}" =~ ^[0-9]+$ ]]
}

jump_host_opt=""
target_user_opt=""
socks_port_opt=""
with_jupyter="false"
jupyter_remote_port_opt=""
jupyter_local_port_opt=""

positionals=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h | --help)
      usage
      exit 0
      ;;
    -J | --jump)
      jump_host_opt="${2:-}"
      [[ -n "$jump_host_opt" ]] || die "--jump requires a value (e.g. user@login)."
      shift 2
      ;;
    -u | --user)
      target_user_opt="${2:-}"
      [[ -n "$target_user_opt" ]] || die "--user requires a value (e.g. myclusteruser)."
      shift 2
      ;;
    -p | --socks-port)
      socks_port_opt="${2:-}"
      [[ -n "$socks_port_opt" ]] || die "--socks-port requires a value (e.g. 1080)."
      shift 2
      ;;
    --with-jupyter)
      with_jupyter="true"
      shift
      ;;
    --jupyter-remote-port | --jupyter-port)
      jupyter_remote_port_opt="${2:-}"
      [[ -n "$jupyter_remote_port_opt" ]] || die "$1 requires a value (e.g. 8888)."
      shift 2
      ;;
    --jupyter-local-port)
      jupyter_local_port_opt="${2:-}"
      [[ -n "$jupyter_local_port_opt" ]] || die "--jupyter-local-port requires a value (e.g. 8888)."
      shift 2
      ;;
    --)
      shift
      positionals+=("$@")
      break
      ;;
    -*)
      die "Unknown option: $1 (use --help)."
      ;;
    *)
      positionals+=("$1")
      shift
      ;;
  esac
done

set -- "${positionals[@]}"

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

# Positional arguments
#
# We support:
#   connection_bridge_tunnel.sh [options] <target> [target_user] [socks_port]
# and, as a convenience for the very common HPC pattern:
#   connection_bridge_tunnel.sh <jump_host> <target_or_url> [target_user] [socks_port]
#
# IMPORTANT: If your jump host is an SSH config name (e.g. "hpc-login") rather than "user@login", use --jump to avoid
# ambiguity with "target user" positional mode.

jump_host=""
target_arg=""
target_user_pos=""
socks_port_pos=""

if [[ -n "$jump_host_opt" ]]; then
  jump_host="$jump_host_opt"
  target_arg="$1"
  shift
else
  if [[ $# -ge 2 ]]; then
    if looks_like_url "$2"; then
      jump_host="$1"
      target_arg="$2"
      shift 2
    elif [[ "$1" == *@* && "$2" != *@* ]]; then
      # Common pattern: "user@login compute123"
      jump_host="$1"
      target_arg="$2"
      shift 2
    else
      target_arg="$1"
      shift
    fi
  else
    target_arg="$1"
    shift
  fi
fi

if [[ $# -ge 1 ]]; then
  target_user_pos="$1"
  shift
fi
if [[ $# -ge 1 ]]; then
  socks_port_pos="$1"
  shift
fi
if [[ $# -gt 0 ]]; then
  die "Too many arguments. Use --help for examples."
fi

# Apply defaults/overrides.
default_user="${USER:-}"
if [[ -z "$default_user" ]]; then
  default_user="$(whoami)"
fi

target_user="${target_user_opt:-${target_user_pos:-$default_user}}"
socks_port="${socks_port_opt:-${socks_port_pos:-1080}}"
jupyter_remote_port="${jupyter_remote_port_opt:-}"
jupyter_local_port="${jupyter_local_port_opt:-}"

if ! is_integer "$socks_port" || ((socks_port < 1 || socks_port > 65535)); then
  die "Invalid socks_port: $socks_port (expected integer 1-65535)."
fi

target_host=""
parsed_jupyter_port=""

if looks_like_url "$target_arg"; then
  target_host="$(echo "$target_arg" | sed -nE 's#^https?://([^/:]+).*#\1#p')"
  parsed_jupyter_port="$(echo "$target_arg" | sed -nE 's#^https?://[^/:]+:([0-9]+).*#\1#p')"
  token="$(echo "$target_arg" | sed -nE 's#.*[?&]token=([^&]+).*#\1#p')"

  if [[ -z "$target_host" || -z "$parsed_jupyter_port" || -z "$token" ]]; then
    die "Could not parse Jupyter URL. Expected: http(s)://<host>:<port>/...token=<...>"
  fi
  case "$target_host" in
    localhost | 127.0.0.1 | 0.0.0.0)
      die "Jupyter URL host is '$target_host', so the compute hostname cannot be inferred. Pass the compute hostname instead."
      ;;
  esac

  if [[ "$with_jupyter" == "true" && -z "$jupyter_remote_port" ]]; then
    jupyter_remote_port="$parsed_jupyter_port"
  fi
else
  # Accept either "host" or "user@host".
  if [[ "$target_arg" == *@* ]]; then
    target_user_in_target="${target_arg%@*}"
    target_host="${target_arg#*@}"
    if [[ -n "${target_user_opt:-}" && "$target_user_opt" != "$target_user_in_target" ]]; then
      warn "Target user specified both via --user ($target_user_opt) and in target ($target_user_in_target); using the one in target."
    fi
    target_user="$target_user_in_target"
  else
    target_host="$target_arg"
  fi
fi

if [[ -z "$target_host" ]]; then
  die "Could not determine target host. Use --help for examples."
fi
if [[ -z "$target_user" ]]; then
  die "Could not determine target user. Use --help for examples."
fi

if [[ "$with_jupyter" == "true" ]]; then
  if [[ -z "$jupyter_remote_port" ]]; then
    die "--with-jupyter requires a Jupyter URL target OR --jupyter-remote-port."
  fi
  if ! is_integer "$jupyter_remote_port" || ((jupyter_remote_port < 1 || jupyter_remote_port > 65535)); then
    die "Invalid --jupyter-remote-port: $jupyter_remote_port (expected integer 1-65535)."
  fi
  if [[ -z "$jupyter_local_port" ]]; then
    jupyter_local_port="$jupyter_remote_port"
  fi
  if ! is_integer "$jupyter_local_port" || ((jupyter_local_port < 1 || jupyter_local_port > 65535)); then
    die "Invalid --jupyter-local-port: $jupyter_local_port (expected integer 1-65535)."
  fi
fi

# Print ONLY the SSH command for easy copy/paste.
cmd=(ssh -N -o ExitOnForwardFailure=yes)
if [[ -n "$jump_host" ]]; then
  cmd+=(-J "$jump_host")
fi
if [[ "$with_jupyter" == "true" ]]; then
  cmd+=(-L "${jupyter_local_port}:127.0.0.1:${jupyter_remote_port}")
fi
cmd+=(-R "127.0.0.1:${socks_port}" "${target_user}@${target_host}")

printf '%s' "${cmd[0]}"
for arg in "${cmd[@]:1}"; do
  printf ' %q' "$arg"
done
printf '\n'
