import pytest
import subprocess
import time
import requests
import os
import signal

@pytest.fixture(scope="session", autouse=True)
def dev_server():
    """Starts the reality-signal-server locally for E2E tests."""

    server_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reality-signal-server"))

    print(f"\nStarting cargo run in {server_dir}...")

    # We use preexec_fn=os.setsid to create a new process group, making it easier to kill the whole tree
    process = subprocess.Popen(
        ["cargo", "run"],
        cwd=server_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        text=True
    )

    # Wait for the server to be ready
    max_retries = 30
    ready = False

    for _ in range(max_retries):
        try:
            # We must use allow_redirects=False because / returns a 301
            response = requests.get("http://localhost:9000/", allow_redirects=False)
            if response.status_code in (200, 301):
                ready = True
                break
        except requests.ConnectionError:
            time.sleep(1)

    if not ready:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        pytest.fail("Server did not start within 30 seconds.")

    print("\nServer is ready at http://localhost:9000/")

    yield process

    print("\nShutting down server...")
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    process.wait(timeout=5)

@pytest.fixture(scope="session")
def browser_type_launch_args(browser_type_launch_args):
    return {
        **browser_type_launch_args,
        "args": [
            "--use-gl=angle",
            "--use-angle=gl",
            "--enable-webgl",
            "--ignore-gpu-blocklist",
            "--enable-features=Vulkan",
            "--enable-unsafe-webgpu"
        ]
    }
