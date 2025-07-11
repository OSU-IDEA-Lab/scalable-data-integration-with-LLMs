import yaml

def get_config() -> dict:
    with open("config.yaml") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)