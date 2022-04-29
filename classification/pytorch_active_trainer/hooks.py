import torch

# dodaj funkcjonalność o uaktualnianiu histogramu co k epok
# dodaj opcje z wyborem warstw
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
        # out = out.mean(axis=0)
        self.writer.log_histogram(f'Pre-Activations/{phase}/{repr(module)}', out.detach())
        
    def gradient_hook(self):
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                module.weight.register_hook(self.grad_hook_wrapper(module))
    
    def grad_hook_wrapper(self, module):
        def grad_hook(grad):
            phase = 'train' if self.model.training else 'val'
            # grad = grad.mean(axis=0)
            self.writer.log_histogram(f'Gradients/{phase}/{repr(module)}', grad.detach())
        return grad_hook

    def weigth_hook(self):
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                pass
                # module.weight.register_hook(lambda : )