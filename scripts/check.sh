#!/bin/sh
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_dir"
# dist is this gate's generated output. Invalidate previous evidence before any
# stage can fail; refuse a symlink/non-directory instead of touching its target.
if [ -L "$project_dir/dist" ] || { [ -e "$project_dir/dist" ] && [ ! -d "$project_dir/dist" ]; }; then
    printf '%s\n' 'dist must be a real generated directory' >&2
    exit 1
fi
rm -rf -- "$project_dir/dist"
gate_context=$(mktemp -d)
gate_token=$(python3 -c 'import uuid; print(uuid.uuid4().hex)')
prepared="forecast-origin-dependencies:$gate_token"
sdk="forecast-origin-sdk:$gate_token"
runtime="forecast-origin-runtime:$gate_token"
sdk_container=''
runtime_container=''
cleanup() {
    if [ -n "$sdk_container" ]; then docker rm -f "$sdk_container" >/dev/null 2>&1 || true; fi
    if [ -n "$runtime_container" ]; then docker rm -f "$runtime_container" >/dev/null 2>&1 || true; fi
    rm -rf -- "$gate_context"
}
trap cleanup EXIT HUP INT TERM
inspect_owned() {
    docker inspect -f '{"image":{{json .Image}},"user":{{json .Config.User}},"network":{{json .HostConfig.NetworkMode}},"binds":{{json .HostConfig.Binds}},"mounts":{{json .Mounts}},"cap_drop":{{json .HostConfig.CapDrop}},"security_opt":{{json .HostConfig.SecurityOpt}},"pids_limit":{{json .HostConfig.PidsLimit}},"memory":{{json .HostConfig.Memory}},"nano_cpus":{{json .HostConfig.NanoCpus}},"read_only":{{json .HostConfig.ReadonlyRootfs}},"exit":{{json .State.ExitCode}},"oom_killed":{{json .State.OOMKilled}}}' "$1"
}
docker build -f Dockerfile.prepare -t "$prepared" .
docker build --network none --build-arg "PREPARED_IMAGE=$prepared" -f Dockerfile -t "$sdk" .
sdk_container=$(docker create --network none --user 1000:1000 --cap-drop ALL --security-opt no-new-privileges --pids-limit 256 --memory 1g --cpus 2 "$sdk")
docker start -a "$sdk_container"
test "$(docker inspect -f '{{.State.ExitCode}}' "$sdk_container")" = 0
mkdir -p dist dist/evidence
docker cp "$sdk_container":/work/distributions/. dist/
docker cp "$sdk_container":/work/evidence/. dist/evidence/
inspect_owned "$sdk_container" > dist/sdk-container.json
docker cp "$sdk_container":/opt/wheelhouse "$gate_context/wheelhouse"
docker cp "$sdk_container":/work/distributions "$gate_context/distributions"
cp requirements-runtime.lock Dockerfile.runtime "$gate_context/"
cp scripts/consumer.py "$gate_context/consumer.py"
cp -R tests/oracles "$gate_context/oracles"
python3 -c 'import json,sys; json.dump(json.load(open("dist/evidence/package-proof.json"))["module_hashes"],open(sys.argv[1],"w"))' "$gate_context/module-hashes.json"
docker build --network none -f "$gate_context/Dockerfile.runtime" -t "$runtime" "$gate_context"
runtime_container=$(docker create --network none --user 1000:1000 --cap-drop ALL --security-opt no-new-privileges --pids-limit 256 --memory 1g --cpus 2 --read-only --tmpfs /tmp:rw,nosuid,nodev,size=32m "$runtime")
docker start -a "$runtime_container"
test "$(docker inspect -f '{{.State.ExitCode}}' "$runtime_container")" = 0
inspect_owned "$runtime_container" > dist/runtime-container.json
sha256sum dist/*.whl dist/*.tar.gz dist/rebuilt/*.whl
printf '%s\n' 'NATIVE_PACKAGE_AND_INSTALLED_CLI_CONSUMER_COMPLETE'
