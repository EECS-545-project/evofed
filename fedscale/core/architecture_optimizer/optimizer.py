from architecture_optimizer.Archi_Manager import Archi_Manager
import yaml
import sys

def optimize(model, opt_times: int):
    conf = None
    with open(sys.path[0]+'/architecture_optimizer/config.yml', 'r') as yf:
        conf = yaml.safe_load(yf)
    manager = Archi_Manager(conf)
    manager.init_agent()
    manager.init_model(model)
    manager.save_model()
    manager.load_policy(opt_times)
    manager.optimize_model(False)
    manager.save_model()
    return manager.model
