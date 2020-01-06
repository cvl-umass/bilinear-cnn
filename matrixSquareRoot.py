import torch
import torch.nn as nn
from torch.autograd import Function

class MatrixSquareRootFun(Function):

    @staticmethod
    def forward(ctx, A, numIters, I, backwardIter):
        bs, dim, _ = A.shape
        normA = A.norm('fro', dim=[1, 2], keepdim=True)
        Y = A.div(normA)

        Z = I.clone()
        Z = Z.unsqueeze(0).repeat(bs, 1, 1)
        I = I.unsqueeze(0).expand(bs, dim, dim)

        for i in range(numIters):
            T = 0.5 * (3.0 * I - Z.bmm(Y))
            Y = Y.bmm(T)
            Z = T.bmm(Z)

        sA = Y.mul(torch.sqrt(normA))
        ctx.save_for_backward(sA, I)
        ctx.backwardIter = backwardIter

        return sA

    @staticmethod
    def backward(ctx, grad_output):
        z = ctx.saved_tensors[0]
        I = ctx.saved_tensors[1]
        bs, dim, _ = z.shape
        normz = z.norm('fro', dim=[1, 2], keepdim=True)
        a = z.div(normz)
        q = grad_output.div(normz)

        for i in range(ctx.backwardIter):
            q = 0.5 * (q.bmm(3.0 * I - a.bmm(a)) - \
                    a.transpose(1, 2).bmm(a.transpose(1,2).bmm(q) - q.bmm(a)))
            a = 0.5 * a.bmm(3.0 * I - a.bmm(a))

        dlda = 0.5 * q
        return (dlda, None, None, None)


class MatrixSquareRoot(nn.Module):
    def __init__(self, numIter, dim, backwardIter=0):
        super(MatrixSquareRoot, self).__init__()
        self.numIter = numIter
        self.dim = dim
        self.register_buffer('I', torch.eye(dim, dim))

        if backwardIter < 1:
            self.backwardIter = numIter
        else:
            self.backwardIter = backwardIter

    def forward(self, x):
        return MatrixSquareRootFun.apply(x, self.numIter, self.I, self.backwardIter)
