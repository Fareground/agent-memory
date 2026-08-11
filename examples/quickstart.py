"""The README hello world, runnable.

Two calls. No lifecycle vocabulary required. The directory it writes is
readable canonical JSON you can commit to git.

Run:  python examples/quickstart.py [memory_dir]
"""

from __future__ import annotations

import sys

from fg_agent_memory import Memory


def main(path: str = "./memory") -> str:
    memory = Memory(path)
    memory.remember("Sandro's favorite editor is Zed.")
    block = memory.recall("what editor does Sandro use?").as_prompt_block()
    return block


if __name__ == "__main__":
    print(main(sys.argv[1] if len(sys.argv) > 1 else "./memory"))
