import torch
import torch.nn.functional as F
import tqdm

class PriorAdamW(torch.optim.AdamW):
    """AdamW optimizer with regularization toward prior parameters
    
    This optimizer performs the standard AdamW update and then applies
    a direct update that pulls parameters toward their prior values,
    weighted by importance (if provided).
    """
    def __init__(self, params, prior_params, model, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, prior_weight=0.0, 
                 importance_weights=None):
        """
        Args:
            params: Model parameters
            prior_params: Dictionary mapping parameter names to their prior values
            model: The model whose parameters are being optimized (used for parameter name mapping)
            lr: Learning rate
            betas: Coefficients for computing running averages of gradient and its square
            eps: Term added to denominator for numerical stability
            weight_decay: Standard AdamW weight decay parameter (L2 toward zero)
            amsgrad: Whether to use the AMSGrad variant
            prior_weight: Weight of regularization toward prior parameters
            importance_weights: Optional dict mapping param names to their importance
                                If None, all parameters are weighted equally
        """
        # Call parent constructor first
        super().__init__(params, lr=lr, betas=betas, eps=eps, 
                         weight_decay=weight_decay, amsgrad=amsgrad)
        
        self.epsilon = eps
        self.prior_params = prior_params
        self.prior_weight = prior_weight
        self.importance_weights = importance_weights or {}
        
        # Build parameter to name mapping
        self.param_to_name_map = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.param_to_name_map[param] = name
    
    def step(self, closure=None):
        """Performs a single optimization step with prior regularization."""
        # Perform standard AdamW update
        loss = super().step(closure)
        
        # Skip if prior_weight is 0
        if self.prior_weight == 0:
            return loss
        
        # Apply regularization toward prior parameters
        with torch.no_grad():
            for group in self.param_groups:
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    
                    # Get parameter name from our mapping
                    param_name = self.param_to_name_map.get(param)
                    
                    # If we have this parameter in prior_params
                    if param_name is not None and param_name in self.prior_params:
                        # Get importance weight (defaults to 1.0 if not specified)
                        importance = self.importance_weights.get(param_name, 1.0)
                        
                        # Apply step toward prior parameter
                        diff = param - self.prior_params[param_name].to(param.device)
                        prior_update = self.prior_weight * importance * diff
                        param.add_(-prior_update)

        return loss

def compute_fisher_matrix(model, data_loader, num_samples=1000, device='cuda'):
    """
    Compute the diagonal of the Fisher Information Matrix for a model.
    
    Args:
        model: PyTorch model
        data_loader: DataLoader containing samples from the task
        num_samples: Maximum number of samples to use (None = use all)
        device: Device to perform computation on
        
    Returns:
        Dictionary mapping parameter names to their Fisher values
    """
    # Set model to evaluation mode
    model.eval()
    
    # Initialize Fisher Information Matrix (diagonal)
    fisher_matrix = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_matrix[name] = torch.zeros_like(param)
    
    # Counter for number of samples processed
    sample_count = 0
    
    # Process data samples
    for batch in tqdm.tqdm(data_loader):
        # Check if we've reached the sample limit
        if num_samples is not None and sample_count >= num_samples:
            break
            
        # For language models, typically input_ids and attention_mask
        if isinstance(batch, dict):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            if 'labels' in batch:
                labels = batch['labels'].to(device)
            else:
                # For autoregressive LMs, labels are typically the input shifted right
                labels = inputs['input_ids'].clone()
        else:
            # Handle tuple/list format
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
        
        # Update sample count
        batch_size = labels.size(0) if hasattr(labels, 'size') else 1
        sample_count += batch_size
        
        # Forward pass
        model.zero_grad()
        outputs = model(**inputs) if isinstance(inputs, dict) else model(inputs)
        
        # For language models, get the logits
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
            
        # For each sample in the batch
        for i in range(batch_size):
            # For language models, we need to handle sequence dimension
            if len(logits.shape) == 3:  # [batch_size, seq_len, vocab_size]
                # Get logits and labels for current sample
                sample_logits = logits[i]
                sample_labels = labels[i] if len(labels.shape) > 1 else labels
                
                # Compute log probabilities
                log_probs = F.log_softmax(sample_logits, dim=-1)
                
                # Sample from the model's distribution
                sampled_indices = torch.multinomial(
                    F.softmax(sample_logits, dim=-1),
                    num_samples=1
                ).squeeze(-1)
                
                # Compute the negative log likelihood of the sampled tokens
                selected_log_probs = log_probs.gather(-1, sampled_indices.unsqueeze(-1)).squeeze(-1)
                loss = -selected_log_probs.sum()
            else:
                # For classification tasks
                sample_logits = logits[i].unsqueeze(0)
                sample_labels = labels[i].unsqueeze(0) if len(labels.shape) > 0 else labels
                
                # Sample from the model's distribution
                probs = F.softmax(sample_logits, dim=-1)
                sampled_class = torch.multinomial(probs, num_samples=1).item()
                
                # Compute loss for the sampled class
                loss = F.cross_entropy(sample_logits, torch.tensor([sampled_class], device=device))
            
            # Backward pass to get gradients
            loss.backward(retain_graph=True)
            
            # Square the gradients and add to Fisher matrix
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_matrix[name] += param.grad.detach().pow(2) / batch_size
            
            # Zero gradients for next iteration
            model.zero_grad()
    
    # Normalize by the number of samples
    for name in fisher_matrix:
        fisher_matrix[name] /= max(1, sample_count)
    
    return fisher_matrix