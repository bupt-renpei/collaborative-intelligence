# Collaborative Intelligence
bottlenet-architecture-torch.ipynb: In this tutorial, we show how we can add a non-differentiable layer (e.g. JPEG compressor/decompressor) in the middle of neural network. An implementation of the bottleneck layer is as follows.
```python
class BottleneckUnit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        result = torch.zeros_like(x)
        batch_size, channels, w, h = x.shape
        outputIoStream = BytesIO()
        image = transforms.ToPILImage()(x.view(batch_size*channels,w*h))
        image.save(outputIoStream, "JPEG", quality=90)
        outputIoStream.seek(0)
        decompressed = Image.open(outputIoStream)
        result = transforms.ToTensor()(decompressed)
        result = result.view(batch_size, channels, w, h)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
```

Related Research:

[1] [BottleNet: A Deep Learning Architecture for Intelligent Mobile Cloud Computing Services](https://ieeexplore.ieee.org/document/8824955)
    Amir Erfan Eshratifar, Amirhossein Esmaili, Massoud Pedram
    IEEE International Symposium on Low Power Electronics and Design, August 2019
    
[2] [JointDNN: An Efficient Training and Inference Engine for Intelligent Mobile Cloud Computing Services](https://ieeexplore.ieee.org/document/8871124)
    Amir Erfan Eshratifar, Mohammad Saeed Abrishami, Massoud Pedram
    IEEE Mobile Computing Journal, October 2019
