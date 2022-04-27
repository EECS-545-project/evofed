import torch
import logging
import math
from torch.autograd import Variable

class Client(object):
    """Basic client component in Federated Learning"""
    def __init__(self, conf):
        pass

    def train(self, client_data, model, conf):

        clientId = conf.clientId
        logging.info(f"Start to train (CLIENT: {clientId}) ...")
        tokenizer, device = conf.tokenizer, conf.device

        model = model.to(device=device)
        model.train()

        trained_unique_samples = min(len(client_data.dataset), conf.local_steps * conf.batch_size)
        global_model = None

        optimizer = torch.optim.SGD(model.parameters(), lr=conf.learning_rate, momentum=0.9, weight_decay=5e-4)

        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device=device)

        epoch_train_loss = 1e-4

        error_type = None
        completed_steps = 0
        loss_squre = 0

        while completed_steps < conf.local_steps:
            
            try:
                for data_pair in client_data:

                    (data, target) = data_pair

                    data = Variable(data).to(device=device)

                    target = Variable(target).to(device=device)

                    output = model(data)
                    loss = criterion(output, target)

                    loss_list = loss.tolist()
                    loss = loss.mean()

                    temp_loss = sum(loss_list)/float(len(loss_list))
                    loss_squre = sum([l**2 for l in loss_list])/float(len(loss_list))
                    # only measure the loss of the first epoch
                    if completed_steps < len(client_data):
                        if epoch_train_loss == 1e-4:
                            epoch_train_loss = temp_loss
                        else:
                            epoch_train_loss = (1. - conf.loss_decay) * epoch_train_loss + conf.loss_decay * temp_loss

                    # ========= Define the backward loss ==============
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    
                    if not conf.use_epoch:
                        completed_steps += 1

                    if completed_steps == conf.local_steps and not conf.use_epoch:
                        break

                completed_steps += 1


            except Exception as ex:
                error_type = ex
                break

        model_param = [param.data.cpu().numpy() for param in model.state_dict().values()]
        results = {'clientId':clientId, 'moving_loss': epoch_train_loss,
                  'trained_size': completed_steps*conf.batch_size, 'success': completed_steps > 0}
        results['utility'] = math.sqrt(loss_squre)*float(trained_unique_samples)

        if error_type is None:
            logging.info(f"Training of (CLIENT: {clientId}) completes, {results}")
        else:
            logging.info(f"Training of (CLIENT: {clientId}) failed as {error_type}")

        results['update_weight'] = model_param
        results['wall_duration'] = 0

        return results


    def test(self, conf):
        pass


