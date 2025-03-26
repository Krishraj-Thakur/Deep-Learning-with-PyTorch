import torch
weights = torch.ones(4, requires_grad=True) #enable tracking of gradients

for epoch in range(3): #running a loop for multiple epochs, 1 in this case, depends on what is in ()
    model_output = (weights*3).sum() #multiplies each value in epoch by 3 and adds them all up

    model_output.backward()

    print(weights.grad)

    weights.grad.zero_() # this is to empty the gradients