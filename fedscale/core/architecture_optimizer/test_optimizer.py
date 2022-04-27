from util.init_model import OptResnet18
model = OptResnet18()
from optimizer import optimize
for key in model.state_dict().keys():
    if "conv" in key:
        print(key)
model = optimize(model, 0)
for key in model.state_dict().keys():
    if "conv" in key:
        print(key)
model = optimize(model, 1)
for key in model.state_dict().keys():
    if "conv" in key:
        print(key)
model = optimize(model, 2)
for key in model.state_dict().keys():
    if "conv" in key:
        print(key)