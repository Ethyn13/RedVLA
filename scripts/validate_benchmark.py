"""Parse benchmark BDDL and rules without running a model or simulator."""
import argparse
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=root / 'configs/examples/suite-routed-client.yml')
    args = parser.parse_args()
    from redvla.config import load_config
    from redvla.safety.parser import get_safety_rules_from_config
    from bddl.parsing import scan_tokens
    _, jobs = load_config(args.config)
    for job in jobs:
        get_safety_rules_from_config(job.task.safety_config_path)
        bddl = next(Path(job.task.scene_dir).glob('*.bddl'))
        tokens = scan_tokens(filename=str(bddl))
        if not tokens or tokens[0] != 'define':
            raise ValueError(f'Invalid BDDL: {bddl}')
    print(f'Parsed scene BDDL and safety rules for {len(jobs)} task entries')


if __name__ == '__main__':
    main()
