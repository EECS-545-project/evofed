import yaml
from Archi_Manager import Archi_Manager
from util.init_model import OptResnet18

if __name__ == '__main__':
    conf = None
    with open('config.yml', 'r') as yf:
        conf = yaml.safe_load(yf)
    
    manager = Archi_Manager(conf)
    initial_model = OptResnet18()
    manager.init_model(initial_model)

    epoch = 0
    log_accuracy = []
    opt = 0

    print('#'*10 + " start training " + '#'*10)
    for i in range(3):
        accu = manager.train_model(50, False)
        manager.save_model()
        print("#"*10 + " stop training " + '#'*10)
        log_accuracy += accu
        print('#'*10 + " start optimizing " + '#'*10)
        manager.optimize_model()
        print('#'*10 + " finish optimizing " + '#'*10)
        manager.save_model()
    print('#'*10 + " start training " + '#'*10)
    accu = manager.train_model(250, False)
    print('#'*10 + " finish training " + '#'*10)
    log_accuracy += accu
    print(log_accuracy)

    
