## EvoFed based on FedScale

This project is larged based on the code base of FedScale with adaptation to support NAS and multiple architectures in Horizontal Federated Learning

### To setup
Please execute `source install.sh` at project root directory to install the dependencies.

Please execute `pip install -e .` at the project root directory under proper virtual environment (if necessary) to install fedscale as a pip package.

### To start experiment
**Basics**:

Go to `evals/` and execute `python manager.py` for help info.

Go to `evals/` and execute `python manager.py submit [config-file-path]` to submit a job distributedly.

Go to `evals/` and execute `python manager.py start [config-file-path]` to start a job locally.

Go to `evals/` and execute `python manager.py stop [job_name]` to terminate a job.

To understand the config file, go to see the comments in it.

**Configs**:

The config file of Evofed is located at `evals/configs/nas/config.yml`

The config file of Heterofl is located at `evals/configs/other/heterofl.yml`

The config file of standard FedAvg is located at `evals/cpu-cifar/config.yml`

**Results**:

You can find your current log file at `evals/job_name_logging`

You can find all your historical log file at `fedscale/logs/job_name/job_start_time`

To visualize the results, you can use the tensorboard, by executing `tensorboard --logdir=[log-file-path]`

**To tune the hyperparameters**:

Just change the value of corrpesponding variable in the config file and start the experiment again.