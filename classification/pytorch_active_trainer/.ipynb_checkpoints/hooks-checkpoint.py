import torch

class Hooks:
    def __init__(self, model, writer):
        self.model = model
        self.writer = writer
        
    def activation_hook(self):
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                module.register_forward_hook(self.act_hook)
                
    def act_hook(self, module, inp, out):
        phase = 'train' if self.model.training else 'val'
        self.writer.add_histogram(f'Pre-Activations/{phase}/{repr(module)}', out)
        
    def gradient_hook(self):
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                module.weight.register_hook(self.grad_hook_wrapper(module))
    
    def grad_hook_wrapper(self, module):
        def grad_hook(grad):
            phase = 'train' if self.model.training else 'val'
            self.writer.add_histogram(f'Gradients/{phase}/{repr(module)}', grad)
        return grad_hook