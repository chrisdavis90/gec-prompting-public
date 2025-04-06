import sys
import subprocess
import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    run_hydra_script = "src/runhydra.py"

    prompt_index = sys.argv[1]
    logger.info(f"Received array index: {prompt_index}")

    # call runhydra.py and pass through system arguments
    args = [
        "python",
        run_hydra_script,
        "prompt_index=" + prompt_index,
    ]
    for arg in sys.argv[2:]:
        if arg.startswith("--"):
            continue
        args.append(arg)

    subprocess.run(args)
