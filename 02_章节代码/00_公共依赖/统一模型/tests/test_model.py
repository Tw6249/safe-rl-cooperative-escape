"""Independent finite differences, identities, and agreement with retained CAP code."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

from thesis_common import (
    ChannelConfig, channel_weight, build_graph, laplacian_rate,
    simple_lambda2_gradient, single_integrator_cbf_rows, row_capacity,
    settings_digest,
)


class CommonModelTests(unittest.TestCase):
    def setUp(self):
        self.config = ChannelConfig.from_settings()

    def test_default_agrees_with_existing_capacity_reserve_model(self):
        core = Path(__file__).resolve().parents[2]
        source = core / "Capacity_Reserve/capacity_reserve_sim/crsim/core.py"
        spec = importlib.util.spec_from_file_location("_cap_reference", source)
        ref = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = ref
        spec.loader.exec_module(ref)
        cfg = ref.ConnectivityConfig()
        for d in [0., .1, .9, 1., 1.2, 1.7, 1.799, 1.8, 2.]:
            with self.subTest(distance=d):
                expected = ref.channel_weight_and_slope(d, cfg)
                np.testing.assert_allclose(channel_weight(d,self.config)[:2], expected, rtol=1e-10,atol=1e-14)

    def test_derivatives_against_independent_finite_differences(self):
        for d in [.2,.8,1.,1.1,1.4,1.7,1.8,1.9]:
            with self.subTest(distance=d):
                # The third derivative jumps at the C2 joins: central error is O(h).
                step=1e-7 if d in (self.config.r_tap,self.config.r_com) else 1e-5
                v,first,second=channel_weight(d,self.config)
                plus=channel_weight(d+step,self.config)
                minus=channel_weight(d-step,self.config)
                self.assertAlmostEqual(first,(plus[0]-minus[0])/(2*step),delta=2e-8)
                self.assertAlmostEqual(second,(plus[1]-minus[1])/(2*step),delta=7e-6)

    def test_cutoff_and_coincident_positions_are_smooth(self):
        value,first,second=channel_weight(0,self.config)
        self.assertEqual(value,1.)
        self.assertEqual(first,0.)
        self.assertAlmostEqual(second,-2/self.config.sigma_w**2)
        self.assertEqual(channel_weight(self.config.r_com,self.config),(0.,0.,0.))
        self.assertEqual(channel_weight(self.config.r_com+1,self.config),(0.,0.,0.))
        p=np.zeros((2,2));g=build_graph(p,self.config)
        np.testing.assert_array_equal(g.gradient_i,0.)
        eps=1e-5;p[0,0]=eps
        self.assertAlmostEqual(build_graph(p,self.config).gradient_i[0,1,0]/eps,second,delta=1e-8)

    def test_strict_monotonicity_and_nonnegative_tail(self):
        distances=np.linspace(0,self.config.r_com,301)
        w,dw,_=channel_weight(distances,self.config)
        self.assertTrue(np.all(w[:-1]>0))
        self.assertTrue(np.all(dw[1:-1]<0))
        self.assertTrue(np.all(np.diff(w)<0))
        near=np.nextafter(self.config.r_com,0.)
        self.assertGreaterEqual(channel_weight(near,self.config)[0],0.)
        # phi(q)=w(sqrt(q)) has negative derivative even at q=0.
        step=1e-7
        dq=(channel_weight(np.sqrt(step),self.config)[0]-1.)/step
        self.assertAlmostEqual(dq,-1/self.config.sigma_w**2,delta=1e-7)

    def test_laplacian_invariants_and_endpoint_gradients(self):
        p=np.array([[0.,0.],[.8,.1],[.2,.7],[2.,.5]])
        g=build_graph(p,self.config)
        np.testing.assert_allclose(g.laplacian.sum(axis=1),0,atol=1e-15)
        np.testing.assert_allclose(g.laplacian,g.laplacian.T)
        self.assertGreaterEqual(np.linalg.eigvalsh(g.laplacian).min(),-1e-14)
        np.testing.assert_allclose(g.gradient_i,-g.gradient_i.transpose(1,0,2))
        for i,j in g.candidate_edges:
            for axis in range(2):
                delta=np.zeros_like(p);delta[i,axis]=1e-6
                finite=(build_graph(p+delta,self.config).adjacency[i,j]-build_graph(p-delta,self.config).adjacency[i,j])/2e-6
                self.assertAlmostEqual(finite,g.gradient_i[i,j,axis],delta=1e-8)

    def test_laplacian_time_derivative(self):
        p=np.array([[0.,0.],[1.2,.3],[.3,.7]])
        u=np.array([[.2,-.1],[-.3,.1],[.1,.2]])
        graph=build_graph(p,self.config)
        eps=1e-6
        finite=(build_graph(p+eps*u,self.config).laplacian-build_graph(p-eps*u,self.config).laplacian)/(2*eps)
        np.testing.assert_allclose(laplacian_rate(graph,u),finite,atol=1e-8)

    def test_gradient_covariance_under_rotation_and_translation(self):
        p=np.array([[0.,0.],[1.1,.1],[.2,.6]])
        rot=np.array([[0.,-1.],[1.,0.]])
        g=build_graph(p,self.config);other=build_graph(p@rot.T+[2.,3.],self.config)
        np.testing.assert_allclose(g.laplacian,other.laplacian,atol=1e-14)
        np.testing.assert_allclose(g.gradient_i@rot.T,other.gradient_i,atol=1e-14)

    def test_simple_spectral_gradient(self):
        p=np.array([[0.,0.],[1.15,.1],[.2,.65]])
        g=build_graph(p,self.config)
        lam,gradient=simple_lambda2_gradient(g,gap_tolerance=1e-9)
        self.assertGreater(lam,0)
        for i in range(3):
            for axis in range(2):
                step=np.zeros_like(p);step[i,axis]=1e-6
                f=lambda x:np.linalg.eigvalsh(build_graph(x,self.config).laplacian)[1]
                self.assertAlmostEqual(gradient[i,axis],(f(p+step)-f(p-step))/2e-6,delta=1e-7)

    def test_repeated_and_disconnected_spectra_are_rejected(self):
        triangle=np.array([[0.,0.],[.8,0.],[.4,np.sqrt(3)*.4]])
        for p in [triangle,np.array([[0.,0.],[4.,0.],[8.,0.]])]:
            with self.assertRaises(ValueError):
                simple_lambda2_gradient(build_graph(p,self.config),gap_tolerance=1e-9)

    def test_cbf_sign_shares_and_zero_sum_allocations(self):
        grad=np.array([[1.,2.],[-3.,.2],[.7,-.1]])
        u=np.array([[.1,.2],[.3,-.1],[-.2,.7]])
        a,b=single_integrator_cbf_rows(grad,h=.4,gain=2.,partial_t_h=-.3,tightening=.1,shares=[.2,.5,.3])
        allocation=np.array([1.,-3.,2.])
        lhs=np.sum(a*u)+np.sum(b+allocation)
        derivative=np.sum(grad*u)-.3
        self.assertAlmostEqual(lhs,-(derivative+2*.4-.1))

    def test_ball_and_box_capacities_are_distinct_support_functions(self):
        a=np.array([3.,4.])
        self.assertEqual(row_capacity(a,2.,input_set="ball_l2"),10.)
        self.assertEqual(row_capacity(a,2.,input_set="box_linf"),14.)
        self.assertEqual(row_capacity(a,[1.,2.],input_set="box_linf"),11.)
        vertices=np.array([[1,2],[1,-2],[-1,2],[-1,-2]])
        self.assertEqual(row_capacity(a,[1.,2.],input_set="box_linf"),np.max(-vertices@a))

    def test_invalid_inputs_do_not_silently_change_the_model(self):
        for args in [(0,1,2,1),(1,2,1,1),(1,1,2,-1),(1,1,np.inf,1)]:
            with self.assertRaises(ValueError):ChannelConfig(*args)
        for d in [-1,np.nan,np.inf]:
            with self.assertRaises(ValueError):channel_weight(d,self.config)
        with self.assertRaises(ValueError):channel_weight(1.,ChannelConfig(.001,.5,2.))
        for edges in [[(0,1),(0,1)],[(1,0)],[(0,0)],[(0,3)],[(0.,1.)]]:
            with self.assertRaises(ValueError):build_graph([[0,0],[1,0]],self.config,edges)
        with self.assertRaises(ValueError):single_integrator_cbf_rows([[1,2],[3,4]],h=1,gain=1,shares=[.4,.4])
        with self.assertRaises(ValueError):row_capacity([1,2],[1,2],input_set="ball_l2")

    def test_settings_identity_and_explicit_sparse_graph(self):
        self.assertEqual(len(settings_digest()),64)
        graph=build_graph([[0.,0.],[.5,0.],[1.,0.]],self.config,[(0,1)])
        self.assertEqual(graph.adjacency[1,2],0.)
        self.assertEqual(graph.candidate_edges,((0,1),))


if __name__=="__main__":unittest.main()
