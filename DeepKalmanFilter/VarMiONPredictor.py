'''
https://github.com/NLALDlab/DeepKalmanFilter/DeepKalmanFilter/VarMiONPredictor.py

E. Chinellato and F. Marcuzzi: State, parameters and hidden dynamics estimation with the Deep Kalman Filter: Regularization strategies. *Journal of Computational Science* **87** (2025), Article number 102569.
10.1016/j.jocs.2025.102569

**SPDX-License-Identifier: GPL-3.0**  
**Copyright (c) 2025 NLALDlab**  
**Authors: Erik Chinellato, Fabio Marcuzzi**
'''
import numpy as np
import torch
from torch import nn

torch.set_default_dtype(torch.float32)

class HeatEquationVarMiONRobin(nn.Module):
    def __init__(self):
        super().__init__()
            

        self.c_branch = nn.Sequential(
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100)           
        )       
        
        self.theta_branch = nn.Sequential(
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100)           
        )

        self.h_branch = nn.Sequential(
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100)           
        )
      
               
        self.f_branch = nn.Sequential(
            nn.Linear(in_features=100, out_features=64)
        )
        
        self.gh_branch = nn.Sequential(
            nn.Linear(in_features=36, out_features=64)
        )
        
        
        
        self.u0_branch = nn.Sequential(
            nn.Linear(in_features=100, out_features=64)         
        )    
        

        self.inversion_net = nn.Sequential(
            nn.Unflatten(1, (1, 10, 10)), 
            nn.ConvTranspose2d(in_channels=1, out_channels=8, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.ConvTranspose2d(in_channels=8, out_channels=16, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(in_channels=16,out_channels= 8, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Unflatten(1,(64,64))
        )

        self.trunk_branch = nn.Sequential(
            nn.Linear(in_features=3, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=64)
        )

    def forward(self, inputs):
        input_tx, input_c, input_theta, input_f, input_h, input_g, input_u0 = inputs
        
        #print("nn forward")
        #print("input_tx", input_tx.size())
        #print("input_c", input_c.size())
        #print("type(input_c): ", type(input_c))
        #print("input_theta", input_theta.size())
        #print("input_f", input_f.size())
        #print("input_h", input_h.size())
        #print("input_g", input_g.size())
        #print("input_u0", input_u0.size())
        
        c_branch = self.c_branch(input_c)
        #print("c_branch ", c_branch.size())
        
        theta_branch = self.theta_branch(input_theta)
        #print("theta_branch ", theta_branch.size())
              
        h_branch = self.h_branch(input_h.view(-1,1).expand(-1, 100))
        #print(f"{h_branch.size() = }")
        
        sum_branch = c_branch + theta_branch + h_branch
        #print(f"{sum_branch.size() = }")
        
        inversion_net = self.inversion_net(sum_branch)
        #print(f"{inversion_net.size() = }")
        
        f_branch = self.f_branch(input_f)
        #print(f"{f_branch.size() = }")
                
        gh_branch = input_h.view(-1,1,1) * self.gh_branch(input_g)
        #print(f"{gh_branch.size() = }")
        
        u0_branch = self.u0_branch(input_u0)
        #print(f"{u0_branch.size() = }")
        
        sum_rhs = f_branch + gh_branch + u0_branch
        #print(f"{sum_rhs.size() = }")
        
        trunk = self.trunk_branch(input_tx)
        #print(f"{trunk.size() = }")
        
        sum_rhs = sum_rhs.unsqueeze(-1)
        #print(f"unsqueeze {sum_rhs.size() = }")
        
        inversion_net = inversion_net.unsqueeze(1)
        #print(f"unsqueeze {inversion_net.size() = }")
        
        output = torch.matmul(inversion_net, sum_rhs)
        #print(f"coeff {output.size() = }")
        
        output = output.unsqueeze(2)
        #print(f"unsqueeze {output.size() = }")
        
        trunk = trunk.unsqueeze(3)
        #print(f"unsqueeze {trunk.size() = }")
        
        output = torch.matmul(trunk, output)
        #print(f"basis {output.size() = }")
        
        output = output.squeeze(-1).squeeze(-1)
        #print(f"squeeze {output.size() = }") 

        return output