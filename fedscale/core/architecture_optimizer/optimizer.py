from multiprocessing import managers
from Archi_Manager import Archi_Manager
import yaml

def optimize(model):
    conf = None
    with open('config.yml', 'r') as yf:
        conf = yaml.safe_load(yf)
    manager = Archi_Manager(conf)
    manager.init_model(model)
    manager.save_model()
    manager.optimize_model()
    manager.save_model()
    return manager.model
