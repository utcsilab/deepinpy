import unittest
import numpy as np
import numpy.testing as npt
import torch

import argparse

from deepinpy.opt import ConjGrad


class TestConjGrad(unittest.TestCase):

    def test_conjgrad_all(self):

        def test_conjgrad(M, N):

            A = np.random.randn(M, N) + 1j * np.random.randn(M, N)
            y = np.random.randn(M, 1) + 1j * np.random.randn(M, 1)

            b = np.conj(A).T.dot(y)

            A_torch = torch.tensor(A)
            b_torch = torch.tensor(b)[None,...]

            def A_normal(x):
                return torch.matmul(torch.conj(A_torch.T), (torch.matmul(A_torch, x)))

            output_np = np.linalg.lstsq(A, y, rcond=None)[0]

            CG_op = ConjGrad(b_torch, A_normal, max_iter=1000, verbose=False)
            output_torch = CG_op(b_torch * 0).numpy().ravel()

            #print(np.stack((abs(output_np.ravel()), abs(output_torch.ravel())), 1))
            nrmse =  np.linalg.norm(output_torch - output_np.ravel()) / np.linalg.norm(output_np.ravel())
            #print('{}x{}: nrmse is:'.format(M, N), nrmse)
            return nrmse

        M_list = [50, 100, 150, 500]
        N_list = [50, 100, 150, 500]

        print('M\tN\tnrmse')
        nrmse_list = []
        for N in N_list:
            for M in M_list:
                nrmse = test_conjgrad(M, N)
                print('{}\t{}\t{:.3e}'.format(M, N, nrmse))
                self.assertTrue(nrmse < 1e-5)


if __name__ == '__main__':
    unittest.main()


