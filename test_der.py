import unittest
import torch
from BCNN import TensorSketch, TensorProduct
import numpy as np
import torch

class TestSketchForward(unittest.TestCase):
    '''
    def test_forward(self):

        D = 100 
        x = torch.rand(2, 20, 1, 1)
        #y = torch.rand(2, 10, 1, 1)
        y = x

        dim_list = [x.shape[1], y.shape[1]]
        bilinear = TensorProduct(dim_list)
        tensor_sketch = TensorSketch(dim_list, D)

        bl1 = bilinear.forward(x, y)
        ts1 = tensor_sketch.forward(x, y)


        diff = torch.dot(ts1[0,:], ts1[1,:]) - torch.dot(bl1[0,:], bl1[1,:]) 
        bl_norm = torch.norm(bl1, dim=1, p=None)

        cos_theta = torch.dot(torch.squeeze(x[0,:]), torch.squeeze(x[1,:])) / \
                    (torch.sqrt(bl_norm[0]) * torch.sqrt(bl_norm[1]))
        eps = torch.sqrt(2 * ((1 / cos_theta)**4)/ (D*1e-2))

        print(diff)
        print(eps * torch.dot(bl1[0,:], bl1[1,:]))
        assert torch.abs(diff) < eps * torch.dot(bl1[0,:], bl1[1,:]) 

        # np.testing.assert_almost_equal(diff.numpy(), 0, decimal=1)

    '''
    def test_forward_diff_res(self):
        x = torch.randn(1, 20, 1, 1)
        y = torch.randn(1, 10, 1, 1)

        rd = torch.randn(1, 10, 1, 1)
        eps = 1e-0

        # x.requires_grad = True
        y.requires_grad = True
        D = 100

        dim_list = [x.shape[1], y.shape[1]]
        tensor_sketch = TensorSketch(dim_list, D)
        # torch.autograd.gradcheck(tensor_sketch.forward, [x, y])

        z1 = torch.sum(torch.squeeze(tensor_sketch.forward(x, y-eps*rd)))
        z2 = torch.sum(torch.squeeze(tensor_sketch.forward(x, y+eps*rd)))

        dL1 = z2 - z1

        z = torch.sum(torch.squeeze(tensor_sketch.forward(x, y)))
        z.backward()
        dL2 = torch.dot(torch.squeeze(y.grad), 2*eps*torch.squeeze(rd))

        import pdb
        pdb.set_trace()
        aaa = 1



if __name__ == '__main__':
    unittest.main()
