#!/bin/bash
# Batched forward-only FFNO service on the GPUs of another host, started over ssh: MPI ranks on CPU nodes (e.g. hyades), GPUs on
# FFNO_FSDP_SERVICE_HOST (e.g. eagle7), which rank 0 also connects to. Pass it as FFNO_FSDP_LAUNCHER. The manifest (in the
# diagnostics directory) and the checkpoints must be on a file system both hosts see. CUDA_VISIBLE_DEVICES, FFNO_FSDP_PYTHON and
# FFNO_BATCHED_* are passed on; FFNO_FSDP_REMOTE_SETUP is run there first (e.g. "source ~/loadnvidia.sh").
set -o errexit
set -o nounset
set -o pipefail

manifest=$(realpath "${1:?usage: ffno_batched_service_ssh.sh MANIFEST.toml}")
host=${FFNO_FSDP_SERVICE_HOST:?set FFNO_FSDP_SERVICE_HOST to the GPU host}
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exports=""
for name in CUDA_VISIBLE_DEVICES FFNO_FSDP_PYTHON $(compgen -e | grep '^FFNO_BATCHED_' || true); do
    [ -n "${!name+x}" ] && exports+="export ${name}=$(printf %q "${!name}"); "
done
command="${FFNO_FSDP_REMOTE_SETUP:-true}; ${exports}exec $(printf %q "${here}/ffno_batched_service_local.sh") $(printf %q "${manifest}")"
echo "Starting the batched FFNO service on ${host} over ssh"
# -tt: when this ssh is killed (the client gave up), the remote service gets SIGHUP instead of being left running.
exec ssh -tt -o BatchMode=yes "${host}" "bash -lc $(printf %q "${command}")"
