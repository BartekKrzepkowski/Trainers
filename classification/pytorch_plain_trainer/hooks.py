import torch

class Hooks:
    def __init__(self, model, writer):
        self.model = model
        self.writer = writer
        
    def forward_hooks(self):
        for module in net.modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                module.register_forward_hook(self.activation_hook)
                
    def activation_hook(self, module, inp, out):
        phase = 'train' if self.model.training else 'val'
        self.writer.add_histogram(f'Pre-Activations/{phase}/{repr(module)}', out)
        
    def gradient_hooks(self):
        for module in net.modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                print(repr(module))
                module.weight.register_hook(self.grad_hook_wrapper(module))
    
    def grad_hook_wrapper(self, module):
        def grad_hook(grad):
            phase = 'train' if self.model.training else 'val'
            self.writer.add_histogram(f'Gradients/{phase}/{repr(module)}', grad)
        return grad_hook