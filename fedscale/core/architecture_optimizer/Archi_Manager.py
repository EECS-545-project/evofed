from unittest import TestLoader
import torch
import copy
import sys
from architecture_optimizer.RL_Agent import RL_Agent
from architecture_optimizer.util.train_model import train
from architecture_optimizer.util.test_model import test
import torchvision
import torchvision.transforms as transforms
from datetime import datetime

class Archi_Manager():
    def __init__(self, conf) -> None:
        self.global_conf = conf
        self.conf = conf["manager"]
        self.batch = self.conf["batch"]
        self.model = None
        self.model_for_train = []
        self.fn_vocab = {}
        self.fn_vocab_back = {}
        for i, l in enumerate(list(range(64, 1088, 16))):
            self.fn_vocab[l] = i
            self.fn_vocab_back[i] = l
        self.fs_vocab = {
            1: 0,
            3: 1,
            5: 2,
            7: 3,
            0: 1,
            1: 3,
            2: 5,
            3: 7
        }
        self.strd_vocab = {
            1: 0,
            2: 1,
            3: 2,
            4: 3
        }
        self.agent = None
        self.num_layers = 16
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=1)
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=1)

    def init_agent(self):
        self.agent = RL_Agent(
            self.global_conf,
            self.num_layers
        )

    def init_model(self, model):
        self.model = copy.deepcopy(model)
        for i in range(self.batch):
            self.model_for_train.append(copy.deepcopy(model))

    def reload_training_model(self):
        self.model_for_train = []
        for i in range(self.batch):
            self.model_for_train.append(copy.deepcopy(self.model))

    def nn2seq(self, is_training: bool=True) -> list:
        """
        convert a torch neural network into a standard layer sequence
        a standard layer sequence has the following vocabulary:
        convolution layer: '[<filter_num>,<filter_size>]'
        by default,
            <filter_num> = [64, 72, ..., 1024]: 64
            <filter_size> = [1, 3, 5, 7]: 4
            <stride> = [1, 2, 3, 4]: 4
            so vocab_size = 64
        output: 
            shape=(batch_size, layer_num, 2)
            batch_size = len(self.model_for_train)
        """
        seq = list()
        if is_training:
            for model in self.model_for_train:
                model_seq = list()
                for i in range(len(model.layers)):
                    for j in range(len(model.layers[i])):
                        for k in range(len(model.layers[i][j].convs)):
                            conv = model.layers[i][j].convs[k]
                            model_seq.append(\
                                self.fn_vocab[conv.out_channels] +\
                                self.fs_vocab[conv.kernel_size[0]]*64 +\
                                self.strd_vocab[conv.stride[0]]*64*4\
                                )
                seq.append(model_seq)
        else:
            model = self.model
            model_seq = list()
            for i in range(len(model.layers)):
                for j in range(len(model.layers[i])):
                    for k in range(len(model.layers[i][j].convs)):
                        conv = model.layers[i][j].convs[k]
                        model_seq.append(\
                            self.fn_vocab[conv.out_channels] +\
                            self.fs_vocab[conv.kernel_size[0]]*64 +\
                            self.strd_vocab[conv.stride[0]]*64*4\
                            )
            seq.append(model_seq)
        
        return torch.tensor(seq)

    def apply_decision(self, wider_decision, deeper_decision, is_training: bool=False) -> None:
        """
        apply the widen or deepen decision on models
        if is_training, apply on self.model_for_train
        else, apply on self.model
        input: 
            wider_decision: [batch, num_layer]
            deeper_decision: [batch, 3]
            widen_decisions is a list of int, denoting the index of widened layer
            deepen_decisions is a list of list of int [layer_index, kernel_size, stride]
        """
        if is_training:
            if wider_decision.shape[0] != len(self.model_for_train):
                raise ValueError(f"Training: wrong number of decisions: {wider_decision.shape[0]} vs {len(self.model_for_train)}")
            for b in range(wider_decision.shape[0]):
                print(f"widen model {b}: #{wider_decision[b]}")
                for i in range(wider_decision.shape[1]):
                    if wider_decision[b,i] == 1:
                        self.model_for_train[b].widden(i)
            for b in range(deeper_decision.shape[0]):
                print(f"deepen model {b}: layer_idx= {deeper_decision[b][0].item()}, filter_size= {self.fs_vocab[deeper_decision[b][1].item()]}, stride={deeper_decision[b][2].item()+1}")
                self.model_for_train[b].deepen(deeper_decision[b][0].item(),\
                                               self.fs_vocab[deeper_decision[b][1].item()],\
                                               deeper_decision[b][2].item()+1)
        else:
            if wider_decision.shape[0] != 1:
                raise ValueError(f"wrong number of decisions: {wider_decision.shape[0]}")
            for i in range(wider_decision.shape[1]):
                print(f"widen model: widen layer #{wider_decision[0]}")
                if wider_decision[0,i] == 1:
                        self.model.widden(i)
            self.model.deepen(deeper_decision[0,0].item(),\
                              self.fs_vocab[deeper_decision[0,1].item()],\
                              deeper_decision[0,2].item()+1)
            print(f"deepen model: layer idx={deeper_decision[0,0].item()}, filter size={self.fs_vocab[deeper_decision[0,1].item()]}, stride={deeper_decision[0,2].item()+1}")
 
    def get_model(self, idx: int, is_training: bool = False):
        if is_training:
            if idx > len(self.model_for_train) - 1:
                raise ValueError(f"wrong index: {idx}")
            return self.model_for_train[idx]
        else:
            return self.model

    def optimize_model(self, need_training: bool=False):
        if need_training:
            self.init_agent()
            for e in range(self.conf["episode"]):
                self.agent.init_trajectory()
                self.reload_training_model()
                for i in range(self.conf["iterations"]):
                    print('#'*10 + f" training optimization policy [{e}, {i}] " + '#'*10)
                    seq = self.nn2seq(True)
                    num_layers = seq.shape[1]
                    wider_decision, deeper_decision = self.agent.sample_decision(seq, num_layers, True, i)
                    self.apply_decision(wider_decision, deeper_decision, True)
                    self.train_model(self.conf["epoch"], True)
                    self.agent.load_reward(self.test_model(True))
                self.agent.update_parameter()
            self.agent.save_policy()
        with torch.no_grad():
            print('#'*10 + f" optimizing model " + '#'*10)
            for i in range(self.conf["iterations"]):
                seq = self.nn2seq(False)
                num_layers = seq.shape[1]
                wider_decision, deeper_decision = self.agent.sample_decision(seq, num_layers, False, i)
                self.apply_decision(wider_decision, deeper_decision, False)
            self.num_layers += self.conf["iterations"]

    
    def train_model(self, epoch: int=10, rl: bool=False):
        if rl:
            for model in self.model_for_train:
                new_model, _ = train(model, epoch=epoch, trainloader=self.trainloader, testloader=self.testloader)
                model.load_state_dict(new_model.state_dict())
        else:
            new_model, accu_log = train(self.model, epoch=epoch, trainloader=self.trainloader, testloader=self.testloader)
            self.model.load_state_dict(new_model.state_dict())
            return accu_log
    
    def test_model(self, rl: bool=False):
        if rl:
            rewards = []
            for i, model in enumerate(self.model_for_train):
                accuracy = test(model, testloader=self.testloader)
                rewards.append(accuracy)
            return rewards
        else:
            accuracy = test(self.model, testloader=self.testloader)
            return accuracy

    def save_model(self):
        timestamp = [datetime.now().day, datetime.now().hour, datetime.now().minute, datetime.now().second]
        timestamp_str = str()
        for time in timestamp:
            timestamp_str += str(time) + '_'
        torch.save(self.model, sys.path[0] + '/architecture_optimizer/' + self.conf["path"] + 'resnet' + timestamp_str + '.pt')

    def load_policy(self, opt_times):
        self.agent.load_policy(opt_times)