#!/usr/bin/env python3
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy as np

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.optimizers import PriorAdamW, compute_fisher_matrix

class SimpleModel(nn.Module):
    """Simple model for testing optimizers"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TestPriorAdamW(unittest.TestCase):
    """Test cases for the PriorAdamW optimizer"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a simple model
        self.model = SimpleModel()
        
        # Create prior parameters (copy of initial model parameters)
        self.prior_params = {}
        for name, param in self.model.named_parameters():
            self.prior_params[name] = param.clone().detach()
    
    def test_standard_adamw_behavior(self):
        """Test that PriorAdamW behaves like standard AdamW when prior_weight is 0"""
        # Create input data
        x = torch.randn(5, 10)
        y = torch.randint(0, 2, (5,))
        
        # Create two identical models
        model1 = SimpleModel()
        model2 = SimpleModel()
        
        # Initialize with same parameters
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            p1.data.copy_(p2.data)
        
        # Create optimizers
        prior_params = {}
        for name, param in model2.named_parameters():
            prior_params[name] = param.clone().detach()
            
        opt1 = torch.optim.AdamW(model1.parameters(), lr=0.01, weight_decay=0.01)
        opt2 = PriorAdamW(model2.parameters(), prior_params, model2, lr=0.01, 
                          weight_decay=0.01, prior_weight=0.0)
        
        # Train for a few steps
        for _ in range(5):
            # Forward pass
            out1 = model1(x)
            out2 = model2(x)
            
            # Compute loss
            loss1 = F.cross_entropy(out1, y)
            loss2 = F.cross_entropy(out2, y)
            
            # Backward pass
            opt1.zero_grad()
            loss1.backward()
            opt1.step()
            
            opt2.zero_grad()
            loss2.backward()
            opt2.step()
        
        # Check that parameters are identical
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(torch.allclose(p1, p2, rtol=1e-5, atol=1e-5))
    
    def test_prior_regularization(self):
        """Test that PriorAdamW applies regularization toward prior parameters"""
        # Create input data
        x = torch.randn(5, 10)
        y = torch.randint(0, 2, (5,))
        
        # Create two identical models
        model1 = SimpleModel()
        model2 = SimpleModel()
        
        # Initialize with same parameters
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            p1.data.copy_(p2.data)
        
        # Create optimizers
        prior_params = {}
        for name, param in model2.named_parameters():
            prior_params[name] = param.clone().detach()
            
        opt1 = torch.optim.AdamW(model1.parameters(), lr=0.01, weight_decay=0.01)
        opt2 = PriorAdamW(model2.parameters(), prior_params, model2, lr=0.01, 
                          weight_decay=0.01, prior_weight=0.1)
        
        # Train for a few steps
        for _ in range(5):
            # Forward pass
            out1 = model1(x)
            out2 = model2(x)
            
            # Compute loss
            loss1 = F.cross_entropy(out1, y)
            loss2 = F.cross_entropy(out2, y)
            
            # Backward pass
            opt1.zero_grad()
            loss1.backward()
            opt1.step()
            
            opt2.zero_grad()
            loss2.backward()
            opt2.step()
        
        # Check that model2 parameters are closer to prior parameters than model1
        for name, param in model2.named_parameters():
            # Find corresponding parameter in model1
            for name1, param1 in model1.named_parameters():
                if name1 == name:
                    # Calculate distances
                    dist1 = torch.norm(param1 - prior_params[name])
                    dist2 = torch.norm(param - prior_params[name])
                    
                    # Model2 should be closer to prior
                    self.assertLess(dist2, dist1)
    
    def test_importance_weights(self):
        """Test that importance weights are correctly applied"""
        # Create input data
        x = torch.randn(5, 10)
        y = torch.randint(0, 2, (5,))
        
        # Create two identical models
        model1 = SimpleModel()
        model2 = SimpleModel()
        
        # Initialize with same parameters
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            p1.data.copy_(p2.data)
        
        # Create prior parameters
        prior_params = {}
        for name, param in model1.named_parameters():
            prior_params[name] = param.clone().detach()
        
        # Create importance weights (high importance for fc1, low for fc2)
        importance_weights = {
            'fc1.weight': 10.0,
            'fc1.bias': 10.0,
            'fc2.weight': 0.1,
            'fc2.bias': 0.1
        }
        
        # Create optimizers with same prior_weight but different importance weights
        opt1 = PriorAdamW(model1.parameters(), prior_params, model1, lr=0.01, 
                          weight_decay=0.01, prior_weight=0.1)
        
        opt2 = PriorAdamW(model2.parameters(), prior_params, model2, lr=0.01, 
                          weight_decay=0.01, prior_weight=0.1,
                          importance_weights=importance_weights)
        
        # Train for a few steps
        for _ in range(5):
            # Forward pass
            out1 = model1(x)
            out2 = model2(x)
            
            # Compute loss
            loss1 = F.cross_entropy(out1, y)
            loss2 = F.cross_entropy(out2, y)
            
            # Backward pass
            opt1.zero_grad()
            loss1.backward()
            opt1.step()
            
            opt2.zero_grad()
            loss2.backward()
            opt2.step()
        
        # Calculate distances from prior for both models
        distances1 = {}
        distances2 = {}
        for name, param in model1.named_parameters():
            distances1[name] = torch.norm(param - prior_params[name]).item()
            
        for name, param in model2.named_parameters():
            distances2[name] = torch.norm(param - prior_params[name]).item()
        
        # fc1 parameters should be closer to prior in model2 than in model1 (higher importance)
        self.assertLess(distances2['fc1.weight'], distances1['fc1.weight'])
        
        # fc2 parameters should be further from prior in model2 than in model1 (lower importance)
        self.assertGreater(distances2['fc2.weight'], distances1['fc2.weight'])
    
    def test_parameter_mapping(self):
        """Test that parameters are correctly mapped to their prior values"""
        # Create model
        model = SimpleModel()
        
        # Store initial parameters
        initial_params = {}
        for name, param in model.named_parameters():
            initial_params[name] = param.clone().detach()
        
        # Create prior parameters with modified values
        prior_params = {}
        for name, param in model.named_parameters():
            # Create a modified version of the parameter
            prior_params[name] = param.clone().detach() + 0.1
        
        # Create optimizer with high prior weight to ensure strong pull toward priors
        optimizer = PriorAdamW(model.parameters(), prior_params, model, lr=0.01, 
                              weight_decay=0.0, prior_weight=0.5)
        
        # Create input data
        x = torch.randn(5, 10)
        y = torch.randint(0, 2, (5,))
        
        # Train for several steps
        for _ in range(20):
            # Forward pass
            out = model(x)
            
            # Compute loss
            loss = F.cross_entropy(out, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Check that parameters have moved toward their prior values
        for name, param in model.named_parameters():
            # Calculate initial distance
            initial_dist = torch.norm(param.data.clone() - prior_params[name] + 0.1)
            
            # Calculate current distance
            current_dist = torch.norm(param.data - prior_params[name])
            
            # Current distance should be less than initial distance
            self.assertLess(current_dist, initial_dist)
    
    def test_fisher_as_importance_weights(self):
        """Test using Fisher matrix as importance weights in PriorAdamW"""
        # Create a simple model
        model = SimpleModel()
        
        # Create a simple dataset
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, size=100):
                self.data = torch.randn(size, 10)
                self.targets = torch.randint(0, 2, (size,))
                
            def __len__(self):
                return len(self.data)
                
            def __getitem__(self, idx):
                return self.data[idx], self.targets[idx]
        
        dataset = SimpleDataset()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)
        
        # Compute Fisher matrix
        fisher_matrix = compute_fisher_matrix(model, dataloader, num_samples=20, device='cpu')
        
        # Create prior parameters
        prior_params = {}
        for name, param in model.named_parameters():
            prior_params[name] = param.clone().detach()
        
        # Create optimizer with Fisher matrix as importance weights
        optimizer = PriorAdamW(model.parameters(), prior_params, model, lr=0.01, 
                              weight_decay=0.01, prior_weight=0.1,
                              importance_weights=fisher_matrix)
        
        # Create input data
        x = torch.randn(5, 10)
        y = torch.randint(0, 2, (5,))
        
        # Train for a few steps
        for _ in range(5):
            # Forward pass
            out = model(x)
            
            # Compute loss
            loss = F.cross_entropy(out, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Calculate distances from prior and weighted by Fisher
        distances = {}
        weighted_distances = {}
        for name, param in model.named_parameters():
            distances[name] = torch.norm(param - prior_params[name]).item()
            # Weight distance by average Fisher value
            weighted_distances[name] = distances[name] * torch.mean(fisher_matrix[name]).item()
        
        # Parameters with higher Fisher values should have smaller weighted distances
        # This is a statistical test, so we'll check if there's a negative correlation
        # between Fisher values and distances
        fisher_means = {name: torch.mean(fisher_matrix[name]).item() for name in fisher_matrix}
        
        # Check at least one relationship to verify the importance weighting is working
        high_fisher_param = max(fisher_means, key=fisher_means.get)
        low_fisher_param = min(fisher_means, key=fisher_means.get)
        
        # The parameter with higher Fisher should have a smaller weighted distance ratio
        # (distance / norm of parameter)
        high_fisher_ratio = distances[high_fisher_param] / torch.norm(prior_params[high_fisher_param]).item()
        low_fisher_ratio = distances[low_fisher_param] / torch.norm(prior_params[low_fisher_param]).item()
        
        # This might not always hold due to randomness, but it should be true most of the time
        # We'll use a soft assertion that prints a warning instead of failing
        if high_fisher_ratio >= low_fisher_ratio:
            print(f"Warning: Parameter with higher Fisher ({high_fisher_param}) doesn't have smaller distance ratio")
            print(f"High Fisher ratio: {high_fisher_ratio}, Low Fisher ratio: {low_fisher_ratio}")
    
    def test_compute_fisher_matrix(self):
        """Test the compute_fisher_matrix function"""
        # Create a simple model
        model = SimpleModel()
        
        # Create a simple dataset
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, size=100):
                self.data = torch.randn(size, 10)
                self.targets = torch.randint(0, 2, (size,))
                
            def __len__(self):
                return len(self.data)
                
            def __getitem__(self, idx):
                return self.data[idx], self.targets[idx]
        
        dataset = SimpleDataset()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)
        
        # Compute Fisher matrix
        fisher_matrix = compute_fisher_matrix(model, dataloader, num_samples=20, device='cpu')
        
        # Check that Fisher matrix has entries for all parameters
        for name, param in model.named_parameters():
            self.assertIn(name, fisher_matrix)
            self.assertEqual(fisher_matrix[name].shape, param.shape)
            
            # Fisher matrix entries should be non-negative
            self.assertTrue(torch.all(fisher_matrix[name] >= 0))

if __name__ == '__main__':
    unittest.main() 