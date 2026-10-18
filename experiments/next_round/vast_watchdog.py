"""Safety net for finish_round.sh: no machine of this code (label nn-*) outlives its run. Every run already destroys
its own machine when it ends; this catches what a crash of the laptop-side process would leave behind.

While the given process lives: every 10 min, destroy nn-* machines older than MAX_AGE_H (a run's own limit is 6 h,
plus copy-back). When it has ended: destroy every nn-* machine. Machines with other labels are never touched.

    nohup uv run python experiments/next_round/vast_watchdog.py <pid> > experiments/next_round/finish/watchdog.log 2>&1 &
"""

import os
import sys
import time

from dotenv import load_dotenv

load_dotenv(".env")
sys.path.insert(0, ".")
from src.train import vast  # noqa: E402

MAX_AGE_H = 7.5


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def ours() -> list[dict]:
    return [i for i in vast.list_instances() if str(i.get("label") or "").startswith(vast.LABEL_PREFIX)]


def main(pid: int) -> None:
    while alive(pid):
        try:
            for inst in ours():
                age_h = (time.time() - (inst.get("start_date") or time.time())) / 3600
                if age_h > MAX_AGE_H:
                    vast.log(f"watchdog: {inst['id']} ({inst.get('label')}) is {age_h:.1f} h old; destroying")
                    vast.destroy_instance(int(inst["id"]))
        except Exception as e:  # API hiccup: try again at the next check
            vast.log(f"watchdog: check failed ({e!r})")
        time.sleep(600)
    vast.log("watchdog: finish_round ended; destroying every nn-* machine left")
    for _ in range(3):
        try:
            left = ours()
            for inst in left:
                vast.log(f"watchdog: destroying {inst['id']} ({inst.get('label')}, {inst.get('actual_status')})")
                vast.destroy_instance(int(inst["id"]))
            if not ours():
                vast.log("watchdog: no nn-* machine left")
                return
        except Exception as e:
            vast.log(f"watchdog: cleanup failed ({e!r})")
        time.sleep(60)


if __name__ == "__main__":
    main(int(sys.argv[1]))
