


########################## Redundant File ##########################





# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import torchvision.transforms as transforms
# import torchvision.datasets

# from darkflow.utils.network_utils import compute_loss


# # Train
# def train_net(model, x_train, wt_train, optimizer, batch_size):
#     input_train = x_train[:, :, :, :].cuda()
#     wt_train = wt_train[:].cuda()
#     model.train()   

#     x_decoded, z_mu, z_var, log_det_j, z0, zk = model(input_train)

#     tr_loss, tr_kl, tr_eucl = compute_loss(input_train, wt_train, x_decoded, z_mu, z_var, batch_size=batch_size)
    
#     # Backprop and perform Adam optimisation
#     optimizer.zero_grad()
#     tr_loss.backward()
#     optimizer.step()

#     return tr_loss, tr_kl, tr_eucl

# # Test/Validate
# def test_net(model, x_test, wt_test, batch_size):
#     model.eval()
#     with torch.no_grad():
#         input_test = x_test[:, :, :, :].cuda()
#         wt_test = wt_test[:].cuda()

#         x_decoded, z_mu, z_var, log_det_j, z0, zk = model(input_test)
        
#         te_loss, te_kl, te_eucl = compute_loss(input_test, wt_test, x_decoded, z_mu, z_var, batch_size=batch_size)

#     return te_loss, te_kl, te_eucl

