"""On-demand Qwen-7B cluster provisioner.

Runs as a Cloud Run service. GitHub Actions POSTs `/spin-up` (authenticated
via Workload Identity Federation) to allocate a spot A100x8 VM, waits for
the eight vLLM replicas to come up, writes the matching `qwen_pool.json`,
and returns its GCS URI. A companion `/tear-down` deletes the VM.

NOT TESTED END-TO-END — needs a GCP project. Run locally with the GCP API
emulator or against a real project. Required env:
  PROJECT_ID, ZONE (e.g. us-central1-a — A100 spot capacity varies; pick a
  region you've successfully preempted in before), POOL_BUCKET (GCS bucket
  for the generated pool JSON), VM_SERVICE_ACCOUNT, IMAGE_FAMILY.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, request
from google.cloud import compute_v1, storage  # type: ignore[import-untyped]

app = Flask(__name__)

PROJECT_ID = os.environ["PROJECT_ID"]
ZONE = os.environ["ZONE"]
POOL_BUCKET = os.environ["POOL_BUCKET"]
VM_SERVICE_ACCOUNT = os.environ["VM_SERVICE_ACCOUNT"]
IMAGE_FAMILY = os.environ.get("IMAGE_FAMILY", "common-cu124-ubuntu-2204-py310")
IMAGE_PROJECT = "deeplearning-platform-release"
MACHINE_TYPE = "a2-highgpu-8g"
ACCELERATOR_TYPE = "nvidia-tesla-a100"
ACCELERATOR_COUNT = 8
MAX_RUN_SECONDS = 60 * 60 * 2   # deadman switch

STARTUP_SCRIPT = (Path(__file__).parent / "startup-script.sh").read_text(encoding="utf-8")


def _build_instance(name: str, hf_token: str) -> compute_v1.Instance:
    boot = compute_v1.AttachedDisk(
        boot=True,
        auto_delete=True,
        initialize_params=compute_v1.AttachedDiskInitializeParams(
            source_image=f"projects/{IMAGE_PROJECT}/global/images/family/{IMAGE_FAMILY}",
            disk_size_gb=200,
        ),
    )
    network_iface = compute_v1.NetworkInterface(
        network=f"projects/{PROJECT_ID}/global/networks/default",
        access_configs=[compute_v1.AccessConfig(name="external", type_="ONE_TO_ONE_NAT")],
    )
    accelerator = compute_v1.AcceleratorConfig(
        accelerator_type=(
            f"projects/{PROJECT_ID}/zones/{ZONE}/acceleratorTypes/{ACCELERATOR_TYPE}"
        ),
        accelerator_count=ACCELERATOR_COUNT,
    )
    scheduling = compute_v1.Scheduling(
        provisioning_model="SPOT",
        preemptible=True,
        on_host_maintenance="TERMINATE",
        instance_termination_action="DELETE",
        max_run_duration=compute_v1.Duration(seconds=MAX_RUN_SECONDS),
    )
    metadata = compute_v1.Metadata(
        items=[
            compute_v1.Items(key="startup-script", value=STARTUP_SCRIPT),
            compute_v1.Items(key="hf-token", value=hf_token),
        ]
    )
    service_account = compute_v1.ServiceAccount(
        email=VM_SERVICE_ACCOUNT,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return compute_v1.Instance(
        name=name,
        machine_type=f"zones/{ZONE}/machineTypes/{MACHINE_TYPE}",
        disks=[boot],
        network_interfaces=[network_iface],
        guest_accelerators=[accelerator],
        scheduling=scheduling,
        metadata=metadata,
        service_accounts=[service_account],
    )


def _wait_for_ready(instances: compute_v1.InstancesClient, name: str, timeout_s: int) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        vm = instances.get(project=PROJECT_ID, zone=ZONE, instance=name)
        if vm.status == "RUNNING":
            return vm.network_interfaces[0].network_i_p   # internal IP
        time.sleep(5)
    raise TimeoutError(f"VM {name} never reached RUNNING within {timeout_s}s")


def _build_pool(internal_ip: str, model: str, run_token: str) -> List[Dict[str, Any]]:
    return [
        {
            "id": f"qwen-{i}",
            "endpoint": f"http://{internal_ip}:{8000 + i}",
            "deployment": model,
            "apiKey": run_token,    # vLLM ignores unless --api-key set; keep for shape
            "apiVersion": "unused",
            "kind": "openai",
        }
        for i in range(8)
    ]


@app.post("/spin-up")
def spin_up() -> Any:
    body = request.get_json(force=True) or {}
    hf_token = body.get("hf_token") or os.environ.get("HF_TOKEN", "")
    model = body.get("model", "Qwen/Qwen2.5-7B-Instruct")
    run_token = uuid.uuid4().hex

    name = f"qwen-bench-{run_token[:10]}"
    instance = _build_instance(name=name, hf_token=hf_token)
    instances = compute_v1.InstancesClient()
    op = instances.insert(project=PROJECT_ID, zone=ZONE, instance_resource=instance)
    op.result(timeout=300)

    internal_ip = _wait_for_ready(instances, name, timeout_s=900)
    pool = _build_pool(internal_ip, model, run_token)

    pool_blob = storage.Client().bucket(POOL_BUCKET).blob(f"{name}/qwen_pool.json")
    pool_blob.upload_from_string(json.dumps(pool, indent=2), content_type="application/json")

    return jsonify(
        {
            "instance": name,
            "pool_uri": f"gs://{POOL_BUCKET}/{name}/qwen_pool.json",
            "internal_ip": internal_ip,
        }
    )


@app.post("/tear-down")
def tear_down() -> Any:
    body = request.get_json(force=True) or {}
    name = body["instance"]
    compute_v1.InstancesClient().delete(project=PROJECT_ID, zone=ZONE, instance=name)
    return jsonify({"deleted": name})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
