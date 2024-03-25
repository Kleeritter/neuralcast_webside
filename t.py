import torch

# check for amd hip
print(torch.cuda.is_available())
print(torch.version.hip)

device = torch.device('cuda')
id = torch.cuda.current_device()
# print gpu name
print(torch.cuda.get_device_name(id))
# no memory is allocated at first
print(torch.cuda.memory_allocated(id))

# store some variable in gpu memory
r = torch.rand(16).to(device)
# memory is allocated
print(torch.cuda.memory_allocated(id))
# crashes when accessing r
print(r[0])