import unittest
import numpy as np

from replica_selection_minerror_v3 import select_cell,select_grid


class SelectionTests(unittest.TestCase):
    def fixture(self):
        b=np.array([[1.],[2.],[4.]])
        boot=np.array([[[.5],[1.9],[3.8]],[[1.5],[2.1],[4.2]],
                       [[.4],[1.8],[3.7]],[[1.6],[2.2],[4.3]]])
        return dict(beta=b,boot=boot,chi=np.array([.1,.2,.3]),
                    chi_boot=np.full((4,3),.2),dof=np.ones(3),
                    penalty=np.ones(3)*2,groups=np.zeros(3),numerical=np.ones(3,bool))

    def test_independent_replica_quality_not_frozen_to_central(self):
        d=self.fixture();d['chi'][1]=1e6
        d['chi_boot'][0]=[1e6,.1,1e6]
        r=select_cell(**d)
        self.assertFalse(r['candidate_quality'][1])
        self.assertTrue(r['candidate_boot_quality'][0,1])
        self.assertAlmostEqual(r['params_boot_selected'][0,0],1.9)
        self.assertFalse(r['replica_fallback_used'][0])

    def test_min_error_not_min_chi2_fallback(self):
        d=self.fixture();d['chi_boot'][0]=[100.,200.,300.]
        r=select_cell(**d)
        self.assertTrue(r['replica_fallback_used'][0])
        self.assertFalse(r['bootstrap_quality_accepted'][0])
        self.assertEqual(r['replica_fallback_candidate_index'][0],1)
        np.testing.assert_allclose(r['stat_error'],r['params_boot_selected'].std(axis=0,ddof=1))

    def test_equal_cut_mixture_and_positive_matching_invariance(self):
        d=self.fixture();d['groups']=np.array([1,1,2])
        r=select_cell(**d)
        self.assertAlmostEqual(r['weights'][:2].sum(),.5)
        self.assertAlmostEqual(r['weights'][2],.5)
        d['beta']*=3;d['boot']*=3
        s=select_cell(**d)
        np.testing.assert_allclose(s['params_selected'],3*r['params_selected'])
        np.testing.assert_allclose(s['weights_boot'],r['weights_boot'])

    def test_tie_break_chi2_then_registered_order(self):
        d=self.fixture();d['boot'][:,:,0]=np.arange(4)[:,None]+np.arange(3)[None]
        d['chi'][:]=[300,100,100]
        r=select_cell(**d)
        self.assertEqual(r['central_fallback_candidate_index'],1)

    def test_zero_dof_last_resort_and_no_fabrication(self):
        d=self.fixture();d['dof'][:]=0
        r=select_cell(**d)
        self.assertFalse(r['selected_available'])
        s=select_cell(**d,allow_zero_dof=True)
        self.assertTrue(s['selected_available'])
        self.assertFalse(s['quality_accepted'])
        self.assertTrue(str(s['central_status']).startswith('zero_dof_'))
        d['boot'][:]=np.nan;d['beta'][:]=np.nan
        self.assertFalse(select_cell(**d,allow_zero_dof=True)['selected_available'])

    def test_grid_axes_and_complete_weights(self):
        d=self.fixture()
        r=select_grid(d['beta'][:,None,None,:],d['boot'][:,:,None,None,:],
            d['chi'][:,None,None],d['chi_boot'][:,:,None,None],d['dof'][:,None,None],
            d['penalty'],d['groups'],d['numerical'][:,None,None])
        self.assertEqual(r['params_boot_selected'].shape,(4,1,1,1))
        self.assertEqual(r['weights_boot'].shape,(4,3,1,1))
        np.testing.assert_allclose(r['weights_boot'].sum(axis=1),1.)
        self.assertTrue(r['bootstrap_selected_available'].all())


if __name__=='__main__':unittest.main()
