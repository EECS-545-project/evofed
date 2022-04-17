import yaml

global cfg
if 'cfg' not in globals():
    with open('evofed/examples/heterofl/config.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)