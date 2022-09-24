import argparse
import sys
import yaml


def read_file(path):
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', dest='base', help='base environment.yml file')
    parser.add_argument('--add', dest='add', help='addon environment.yml file')
    return parser.parse_args(argv)


def merge_dependencies(env_base, env_add):
    base_dict = {f.split()[0]: f for f in env_base}
    add_dict = {f.split()[0]: f for f in env_add}
    for k,v in add_dict.items():
        if k not in base_dict.keys():
            base_dict[k] = v
    return list(base_dict.values())


def merge_channels(env_base, env_add):
    for c in env_add:
        if c not in env_base:
            env_base.append(c)
    return env_base


def merge_env(env_base, env_add):
    return {
        "channels": merge_channels(
            env_base=env_base['channels'], 
            env_add=env_add['channels']
        ),
        'dependencies': merge_dependencies(
            env_base=env_base['dependencies'], 
            env_add=env_add['dependencies']
        )
    }


if __name__ == '__main__':
    arguments = parse_args(argv=None)
    yaml.dump(
        merge_env(
            env_base=read_file(arguments.base), 
            env_add=read_file(arguments.add)
        ),
        sys.stdout, 
        indent=2, 
        default_flow_style=False
    )
