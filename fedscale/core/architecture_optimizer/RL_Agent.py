from datetime import datetime
import torch
import torch.optim as optim
from Policy_Network import Policy_Network, Wider_Actor

class RL_Agent():
    def __init__(self, conf, num_layers: int=16) -> None:
        self.network = None
        self.episode = 0
        self.conf = conf
        self.iterations = conf["manager"]["iterations"]
        self.reward = [] # [iteration, bash]
        self.batch = conf["manager"]["batch"]
        self.num_layers = num_layers
        self.policy = Policy_Network(
            conf["rl"]["encoder_vocab_size"],
            conf["rl"]["encoder_embedding_dim"],
            conf["rl"]["encoder_hidden_dim"],
            conf["rl"]["deeper_embedding_dim"],
            [self.num_layers, 4, 4],
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.conf["rl"]["lr"])
        self.deeper_decision_trajectory = [torch.zeros((self.batch, 3), dtype=torch.int32)] # stack of shape: (batch, 3)
        self.wider_decision_trajectory = [torch.zeros((self.batch, self.num_layers), dtype=torch.int32)]  # stack of shape: (batch, num_layers)
        self.deeper_probs_trajectory = list()             # stack of shape: len(out_dims), batch, out_dims[]
        self.wider_probs_trajectory = list()              # stack of shape: (batch, num_layers, 2)
    
    def init_trajectory(self):
        self.deeper_decision_trajectory = [torch.zeros((self.batch, 3), dtype=torch.int32)] # stack of shape: (batch, 3)
        self.wider_decision_trajectory = [torch.zeros((self.batch, self.num_layers), dtype=torch.int32)]  # stack of shape: (batch, num_layers)
        self.deeper_probs_trajectory = list()             # stack of shape: len(out_dims), batch, out_dims[]
        self.wider_probs_trajectory = list()              # stack of shape: (batch, num_layers, 2)
        self.reward = []

    def sample_decision(self, input, num_layers, is_training: bool, iteration: int):
        if is_training:
            self.optimizer.zero_grad()
            self.policy.train(True)
        else:
            self.policy.train(False)
        deeper_decision, deeper_probs, wider_decision, wider_probs \
            = self.policy(input, num_layers, is_training, self.deeper_decision_trajectory[iteration])
        if is_training:
            self.wider_decision_trajectory.append(wider_decision)
            self.deeper_decision_trajectory.append(deeper_decision)
            self.deeper_probs_trajectory.append(deeper_probs)
            self.wider_probs_trajectory.append(wider_probs)
        return wider_decision, deeper_decision

    def update_parameter(self):
        print(f"policy reward: {self.reward}")
        reward = torch.tensor(self.reward)
        loss = 0
        for i in range(self.batch):
            for t in range(self.iterations):
                ret = sum(reward[t:,i])
                wider_probs_i = torch.log(self.wider_probs_trajectory[t][i])
                wider_decision = self.wider_decision_trajectory[t+1][i].to(torch.long)
                wider_decision_mask = torch.eye(wider_probs_i.shape[1])[wider_decision]
                wider_entropy = torch.sum(torch.mul(wider_probs_i, wider_decision_mask))
                deeper_probs = self.deeper_probs_trajectory[t]
                deeper_decision_i = self.deeper_decision_trajectory[t+1][i]

                deeper_entropy = 0
                for k in range(3):
                    deeper_entropy += torch.log(deeper_probs[k][i][deeper_decision_i[k]])

                loss += -(deeper_entropy + wider_entropy) * ret / self.batch
        loss.backward(retain_graph=True)
        self.optimizer.step()
        print(f"policy loss: {loss}")

    def load_reward(self, reward):
        self.reward.append(reward)
    
    def save_policy(self):
        timestamp = [datetime.now().day, datetime.now().hour, datetime.now().minute, datetime.now().second]
        timestamp_str = str()
        for time in timestamp:
            timestamp_str += str(time) + '_'
        torch.save(self.policy, self.conf["path"] + 'policy' + timestamp_str + '.pt')