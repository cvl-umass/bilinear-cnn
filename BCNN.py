import feature_extractor as fe
import torchvision
import torch
import torch.nn as nn
import functools
import operator
from compact_bilinear_pooling import CountSketch
from torch.autograd import Function
from matrixSquareRoot import MatrixSquareRoot 
import torch.nn.functional as F

matrix_sqrt = MatrixSquareRoot.apply

def create_backbone(model_name, finetune_model=True, use_pretrained=True):
    model_ft = None
    input_size = 0

    if model_name == 'vgg':
        """ VGG
        """
        model_ft = fe.VGG() 
        set_parameter_requires_grad(model_ft, finetune_model)
        
        output_dim = 512

    elif model_name == "resnet":
        """ Resnet101
        """
        model_ft = fe.ResNet()
        set_parameter_requires_grad(model_ft, finetune_model)
        # num_ftrs = model_ft.fc.in_features
        # model_ft.fc = nn.Linear(num_ftrs, num_classes)
        # input_size = 224

        output_dim = 2048

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = fe.DenseNet()
        set_parameter_requires_grad(model_ft, finetune_model)
        # num_ftrs = model_ft.classifier.in_features
        # model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        # input_size = 224

        output_dim = 1920

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = fe.Inception()
        set_parameter_requires_grad(model_ft, finetune_model)
        
        output_dim = 2048
    else:
        # print("Invalid model name, exiting...")
        # logger.debug("Invalid mode name")
        exit()

    return model_ft, output_dim

def set_parameter_requires_grad(model, requires_grad):
    if requires_grad:
        for param in model.parameters():
            param.requires_grad = True

# This implementation only works for VGG networks 
class MultiHeadsBCNN(nn.Module):
    def __init__(self, num_classes, feature_extractors=None):
        super(MultiHeadsBCNN, self).__init__()
        self.feature_extractors = feature_extractors
        dim_all_layers = feature_extractors.get_feature_dims()
        self.pooling_fn_list = nn.ModuleList(
                [TensorProduct([dim] * 2) for dim in dim_all_layers]
        )
        self.feature_dim_list = [
                pooling_fn.get_output_dim()
                for pooling_fn in self.pooling_fn_list
        ]
        self.fc_list = nn.ModuleList(
                [nn.Linear(feature_dim, num_classes, bias=True)
                for feature_dim in self.feature_dim_list]
        )

    def forward(self, x):
        relu_acts = self.feature_extractors(x)

        bs, _, h1, w1 = x.shape

        bcnn_list = [
                pooling_fn(z, z)
                for z, pooling_fn in zip(relu_acts, self.pooling_fn_list)
        ]
        bcnn_list = [
                z.view(bs, feature_dim)
                for z, feature_dim in zip(bcnn_list, self.feature_dim_list)
        ]

        bcnn_list = [
                torch.sqrt(F.relu(z) + 1e-5) - torch.sqrt(F.relu(-z) + 1e-5)
                for z in bcnn_list
        ]
        
        bcnn_list = [
                torch.nn.functional.normalize(z) for
                z in bcnn_list
        ]

        y = [fc(z) for z, fc in zip(bcnn_list, self.fc_list)]

        return y


class BCNNModule(nn.Module):
    def __init__(self, num_classes, feature_extractors=None,
            pooling_fn=None, order=2, m_sqrt_iter=0, demo_agg=False,
            fc_bottleneck=False, learn_proj=False):
        super(BCNNModule, self).__init__()

        assert feature_extractors is not None
        assert pooling_fn is not None

        self.feature_extractors = feature_extractors
        self.pooling_fn = pooling_fn 

        self.feature_dim = self.pooling_fn.get_output_dim()
        if fc_bottleneck:
            self.fc = nn.Sequential(nn.Linear(self.feature_dim, 1024, bias=True), 
                                    nn.Linear(1024, num_classes, bias=True))
        else:
            self.fc = nn.Linear(self.feature_dim, num_classes, bias=True) 

        # TODO assert m_sqrt is not used together with tensor sketch nor
        # the BCNN models without sharing
        if m_sqrt_iter > 0:
            self.m_sqrt = MatrixSquareRoot(
                m_sqrt_iter,
                int(self.feature_dim ** 0.5),
                backwardIter=5
            )
        else:
            self.m_sqrt = None

        self.demo_agg = demo_agg
        self.order = order
        self.learn_proj = learn_proj

    def get_order(self):
        return self.order

    def forward(self, *args):
        x = self.feature_extractors(*args)

        bs, _, h1, w1 = x[0].shape
        for i in range(1, len(args)):
            _, _, h2, w2 = x[i].shape
            if h1 != h2 or w1 != w2:
                x[i] = torch.nn.functional.interpolate(x[i], size=(h1, w1),
                                                    mode='bilinear')
        z = self.pooling_fn(*x)

        # TODO improve coding style, modulize normlaization operations
        #      use a list of normalization operations
        # normalization

        if self.m_sqrt is not None:
            z = self.m_sqrt(z)
        z = z.view(bs, self.feature_dim)
        z = torch.sqrt(F.relu(z) + 1e-5) - torch.sqrt(F.relu(-z) + 1e-5)
        z = torch.nn.functional.normalize(z)

        # linear classifier
        y = self.fc(z)

        return y


class MultiStreamsCNNExtractors(nn.Module):
    def __init__(self, backbones_list, dim_list, proj_dim=0):
        super(MultiStreamsCNNExtractors, self).__init__()
        self.feature_extractors = nn.ModuleList(backbones_list)
        if proj_dim > 0:
            temp = [nn.Sequential(x, \
                        nn.Conv2d(fe_dim, proj_dim, 1, 1, bias=False)) \
                        for x, fe_dim in zip(self.feature_extractors, dim_list)]
            self.feature_extractors = nn.ModuleList(temp)

class BCNN_sharing(MultiStreamsCNNExtractors):
    def __init__(self, backbones_list, dim_list, proj_dim=0, order=2):
        super(BCNN_sharing, self).__init__(backbones_list, dim_list, proj_dim)

        # one backbone network for sharing parameters
        assert len(backbones_list) == 1 

        self.order = order

    def get_number_output(self):
        return self.order

    def forward(self, *args):
        # y = self.feature_extractors[0](x)
        y = [self.feature_extractors[0](x) for x in args]

        if len(args) == 1:
            # out = y * self.order
            # y[0].register_hook(lambda grad: print(grad[0,0,:3,:3]))

            # return out
            return y * self.order
            # return [y for z in range(self.order)] 
        else:
            return y

class BCNN_no_sharing(MultiStreamsCNNExtractors):
    def __init__(self, backbones_list, dim_list, proj_dim=0):
        super(BCNN_no_sharing, self).__init__(backbones_list, dim_list, proj_dim)

        # two networks for the model without sharing
        assert len(backbones_list) >= 2
        self.order = len(backbones_list)

    def get_number_output(self):
        return self.order

    def forward(self, *args):
        y = [fe(x) for x, fe in zip(args, self.feature_extractors)]

        return y

class TensorProduct(nn.Module):
    def __init__(self, dim_list):
        super(TensorProduct, self).__init__()
        self.output_dim = functools.reduce(operator.mul, dim_list)

        # Use tensor sketch for the order greater than 2
        assert len(dim_list) == 2

    def get_output_dim(self):
        return self.output_dim

    def forward(self, *args):
        (x1, x2) = args
        [bs, c1, h1, w1] = x1.size()
        [bs, c2, h2, w2] = x2.size()
        
        x1 = x1.view(bs, c1, h1*w1)
        x2 = x2.view(bs, c2, h2*w2)
        y = torch.bmm(x1, torch.transpose(x2, 1, 2))

        # return y.view(bs, c1*c2) / (h1 * w1)
        return y / (h1 * w1)


class TensorSketch(nn.Module):
    def __init__(self, dim_list, embedding_dim=4096, pooling=True,
                    update_sketch=False):
        super(TensorSketch, self).__init__()


        self.output_dim = embedding_dim 

        self.count_sketch = nn.ModuleList(
                    [CountSketch(dim, embedding_dim, update_proj=update_sketch) \
                            for dim in dim_list])
        self.pooling = pooling 

    def get_output_dim(self):
        return self.output_dim

    def forward(self, *args):
        y = [sketch_fn(x.permute(0,2,3,1)) \
                for x, sketch_fn in zip(args, self.count_sketch)] 

        z = ApproxTensorProduct.apply(self.output_dim, *y)
        _, h, w, _ = z.shape

        if self.pooling:
            return torch.squeeze(
                    torch.nn.functional.avg_pool2d(z.permute(0,3,1,2), (h, w)))
        else:
            return z.permute(0, 3, 1, 2)
        
class SketchGammaDemocratic(nn.ModuleList):
    def __init__(self, dim_list, embedding_dim=4096,
                gamma=0, sinkhorn_t=0.5, sinkhorn_iter=10, update_sketch=False):
        super(SketchGammaDemocratic, self).__init__()
        self.sketch = TensorSketch(dim_list, embedding_dim, False, update_sketch)
        output_dim = self.sketch.get_output_dim()
        self.gamma_demo = GammaDemocratic(output_dim, gamma, sinkhorn_t, sinkhorn_iter)

    def forward(self, *args):
        x = self.sketch(*args) 
        x = self.gamma_demo(x)

        return x

    def get_output_dim(self):
        return self.sketch.get_output_dim()

class GammaDemocratic(nn.ModuleList):
    def __init__(self, output_dim, gamma=0, sinkhorn_t=0.5, sinkhorn_iter=10):
        super(GammaDemocratic, self).__init__()
        self.sinkhorn_t = sinkhorn_t    # dampening parameter
        self.gamma = gamma
        self.sinkhorn_iter = sinkhorn_iter
        self.output_dim = output_dim 

    def forward(self, x):
        [bs, ch, h, w] = x.shape
        x = x.view(bs, ch, -1).transpose(2, 1)
        # x.register_hook(self.save_grad('x'))

        K = x.bmm(x.transpose(2, 1))
        K = (K + torch.abs(K)) / 2

        # alpha = torch.autograd.Variable(torch.ones(bs, h*w, 1)).cuda()
        alpha = torch.ones_like(x[:,:,[0]]) 
        Ci = torch.sum(K, 2, keepdim=True)
        Ci = torch.pow(Ci, self.gamma).detach()

        for _ in range(self.sinkhorn_iter):
            # alpha = torch.pow(alpha + 1e-10, 1-self.sinkhorn_t) * \
            #         torch.pow(Ci + 1e-10, self.sinkhorn_t) / \
            #         (torch.pow(K.bmm(alpha) + 1e-10, self.sinkhorn_t) + 1e-10)
            alpha = torch.pow(Ci + 1e-10, self.sinkhorn_t) * \
                    torch.pow(alpha + 1e-10, 1-self.sinkhorn_t) / \
                    (torch.pow(K.bmm(alpha) + 1e-10, self.sinkhorn_t) + 1e-10)

        x = torch.sum(x * alpha, dim=1, keepdim=False)

        return x

    def get_output_dim(self):
        return self.output_dim

class SecondOrderGammaDemocratic(nn.Module):
    def __init__(self, output_dim, gamma=0, sinkhorn_t=0.5, sinkhorn_iter=10):
        super(SecondOrderGammaDemocratic, self).__init__()
        self.sinkhorn_t = sinkhorn_t    # dampening parameter
        self.sinkhorn_iter = sinkhorn_iter
        self.gamma = gamma
        self.iter = sinkhorn_iter
        # self.grad = {}
        self.output_dim = output_dim

    def forward(self, *args):
        # The forward assume args[0] == args[1]. This should be asserted during
        # model creation

        x = args[0]
        [bs, ch, h, w] = x.shape
        x = x.view(bs, ch, -1).transpose(2, 1)

        K = x.bmm(x.transpose(2, 1))
        K = K * K;

        alpha = torch.ones_like(x[:,:,[0]]) 
        Ci = torch.sum(K, 2, keepdim=True)
        Ci = torch.pow(Ci, self.gamma).detach()

        for _ in range(self.sinkhorn_iter):
            alpha = torch.pow(Ci + 1e-10, self.sinkhorn_t) * \
                    torch.pow(alpha + 1e-10, 1-self.sinkhorn_t) / \
                    (torch.pow(K.bmm(alpha) + 1e-10, self.sinkhorn_t) + 1e-10)

        x = x * torch.pow(alpha + 1e-8, 0.5)
        x = x.transpose(1, 2).bmm(x).view(bs, -1)

        return x

    def get_output_dim(self):
        return self.output_dim


class ApproxTensorProduct(Function):

    @staticmethod
    def forward(ctx, embedding_dim, *args):
        fx = [torch.rfft(x, 1) for x in args]

        re_fx1 = fx[0].select(-1, 0) 
        im_fx1 = fx[0].select(-1, 1)
        for i in range(1, len(fx)):
            re_fx2 = fx[i].select(-1, 0)
            im_fx2 = fx[i].select(-1, 1)

            # complex number multiplication
            Z_re = torch.addcmul(re_fx1*re_fx2, -1, im_fx1, im_fx2)
            Z_im = torch.addcmul(re_fx1*im_fx2,  1, im_fx1, re_fx2)
            re_fx1 = Z_re
            im_fx1 = Z_im

        ctx.save_for_backward(re_fx1, im_fx1, *fx)
        # ctx.save_for_backward(*fx)
        re = torch.irfft(torch.stack((re_fx1, im_fx1), re_fx1.dim()), 1,
                        signal_sizes=(embedding_dim,))

        ctx.embedding_dim = embedding_dim
        return re

    @staticmethod
    def backward(ctx, grad_output):

        grad_output = grad_output.contiguous()
        grad_prod = torch.rfft(grad_output, 1)
        grad_re_prod = grad_prod.select(-1, 0)
        grad_im_prod = grad_prod.select(-1, 1)

        re_fout = ctx.saved_tensors[0]
        im_fout = ctx.saved_tensors[1]

        fx = ctx.saved_tensors[2:]
        grad = []
        for fi in fx:
            re_fi = fi.select(-1, 0)
            im_fi = fi.select(-1, 1)

            temp_norm = (re_fi**2 + im_fi**2 + 1e-8)
            temp_re = torch.addcmul(re_fout * re_fi, 1, im_fout, im_fi) \
                        / temp_norm
            temp_im = torch.addcmul(im_fout * re_fi, -1, re_fout, im_fi) \
                        /temp_norm
            grad_re = torch.addcmul(grad_re_prod * temp_re, 1,
                            temp_im, grad_im_prod)
            grad_im = torch.addcmul(grad_im_prod * temp_re, -1,
                            grad_re_prod, temp_im)
            grad_fi = torch.irfft(
                    torch.stack((grad_re, grad_im), grad_re.dim()), 1,
                    signal_sizes=(ctx.embedding_dim,))
            grad.append(grad_fi)

        return (None, *grad)


def create_bcnn_model(model_names_list, num_classes,
                pooling_method='outer_product', fine_tune=True, pre_train=True,
                embedding_dim=8192, order=2, m_sqrt_iter=0,
                fc_bottleneck=False, proj_dim=0, update_sketch=False,
                gamma=0.5):

    temp_list = [create_backbone(model_name, finetune_model=fine_tune, \
            use_pretrained=pre_train) for model_name in model_names_list]

    temp_list = list(map(list, zip(*temp_list)))
    backbones_list = temp_list[0]

    # list of feature dimensions of the backbone networks
    dim_list = temp_list[1]
    
    # BCNN mdoels with sharing parameters. The computation of the two backbone
    # networks are shared resulting in a symmetric BCNN
    if len(backbones_list) == 1:
        assert order >= 2
        dim_list = dim_list * order
        feature_extractors = BCNN_sharing(
                backbones_list,
                dim_list,
                proj_dim, order
        )
    else:
        feature_extractors = BCNN_no_sharing(backbones_list, dim_list, proj_dim)

    # update the reduced feature dimension in dim_list if there is
    # dimensionality reduction
    if proj_dim > 0:
        dim_list = [proj_dim for x in dim_list]

    if pooling_method == 'outer_product':
        pooling_fn = TensorProduct(dim_list)
    elif pooling_method == 'sketch':
        pooling_fn = TensorSketch(dim_list, embedding_dim, True, update_sketch)
    elif pooling_method == 'gamma_demo':
        assert isinstance(feature_extractors, BCNN_sharing) 
        pooling_fn = SecondOrderGammaDemocratic(dim_list[0] ** 2, gamma=gamma, sinkhorn_t=0.5,
                                                sinkhorn_iter=10)
    elif pooling_method == 'sketch_gamma_demo':
        pooling_fn = SketchGammaDemocratic(
                dim_list,
                embedding_dim,
                gamma=gamma,
                sinkhorn_t=0.5,
                sinkhorn_iter=10,
                update_sketch=update_sketch
        )
    else:
        raise ValueError('Unknown pooling method: %s' % pooling_method)

    learn_proj = True if proj_dim > 0 else False
    return BCNNModule(
            num_classes,
            feature_extractors,
            pooling_fn,
            order,
            m_sqrt_iter=m_sqrt_iter,
            fc_bottleneck=fc_bottleneck,
            learn_proj=learn_proj
    )

def create_multi_heads_bcnn(num_classes):

    backbone = fe.VGG_all_conv_features()
    return MultiHeadsBCNN(num_classes, backbone)

