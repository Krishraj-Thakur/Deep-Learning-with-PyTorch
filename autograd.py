import torch

x = torch.randn(3, requires_grad=True)
print(x)

y = x + 2
print(y)
z = y*y*2
#z = z.mean()
print(z)

v = torch.tensor([0.1,1.0,0.001], dtype=torch.float32)
z.backward(v) #dz/dx
print(x.grad)

#3 ways stop pytorch from creating gradient functions and tracking the history in the computational graph
#x.requires_grad_false()
#x.detach()
#with torch.no_grad():