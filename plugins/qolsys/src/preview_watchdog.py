"""Independent native stop if the preview worker disappears.

Retries until our recording has ended, because stopping is deferred while an
alarm owns the panel camera.
"""
import asyncio
import contextlib
import sys
import time
from pathlib import Path

from panel_client import PanelClient

RETRY_SECONDS = 15
GIVE_UP_SECONDS = 3600


async def main():
    directory, panel, local, remote, delay, request_id, resume_motion = sys.argv[1:]
    await asyncio.sleep(float(delay))
    client = PanelClient(Path(directory), panel, local, remote)
    deadline = time.monotonic() + GIVE_UP_SECONDS
    try:
        while time.monotonic() < deadline:
            with contextlib.suppress(Exception):
                if (await client.stop_recording(request_id, resume_motion == "true")
                        or await client.recording_finished(request_id)):
                    return
            await asyncio.sleep(RETRY_SECONDS)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
