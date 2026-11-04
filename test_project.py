"""Meaningful pricing, sensitivity, constraint, and no-lookahead checks."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import yield_curve_project as project


class ResearchChecks(unittest.TestCase):
    def setUp(self):
        self.flat = np.array([5.,0.,0.,0.,1.,2.])

    def test_flat_curve_and_zero_maturity_limit(self):
        np.testing.assert_allclose(project.nss_zero_rate([0,1,5],self.flat),.05)
        params=np.array([3.,-1.,2.,1.,1.,2.])
        self.assertAlmostEqual(float(project.nss_zero_rate(0,params)),.02)

    def test_zero_coupon_analytical_price(self):
        bond=project.Bond(5,coupon=0)
        self.assertAlmostEqual(project.price_bond(bond,self.flat),100*np.exp(-.25))

    def test_coupon_bond_par_when_conventions_match(self):
        bond=project.Bond(5,coupon=.04)
        rate=2*np.log(1+.04/2)
        params=np.array([rate*100,0,0,0,1,2])
        self.assertAlmostEqual(project.price_bond(bond,params),100)
        self.assertAlmostEqual(project.yield_to_maturity(bond,100),.04)

    def test_duration_convexity_against_finite_differences(self):
        bond=project.Bond(5)
        jac,d,c=project.sensitivities([bond],self.flat)
        bump=1e-5
        p=project.price_bond(bond,self.flat)
        up=project.price_bond(bond,self.flat,parallel_shift=bump)
        down=project.price_bond(bond,self.flat,parallel_shift=-bump)
        self.assertAlmostEqual(d[0],-(up-down)/(2*bump*p),places=7)
        self.assertAlmostEqual(c[0],(up+down-2*p)/(bump*bump*p),places=4)
        self.assertAlmostEqual(-jac.sum(),d[0])

    def test_aging_and_coupon_receipts(self):
        bond=project.Bond(1,coupon=.04)
        p=project.price_bond(bond,self.flat)
        # At exactly six months the first coupon becomes cash, not remaining PV.
        expected=(102*np.exp(-.05*.5)+2)/p-1
        self.assertAlmostEqual(project.holding_return(bond,self.flat,self.flat,.5),expected)
        self.assertAlmostEqual(project.holding_return(bond,self.flat,self.flat,1),104/p-1)

    def test_risk_models_are_positive_semidefinite(self):
        # Deterministic correlated observations, no random simulation.
        t=np.arange(80)
        history=np.column_stack([np.sin(t/7+j)*.001 for j in np.arange(20)/20])
        jac,d,_=project.sensitivities([project.Bond(m) for m in project.MATURITIES],self.flat)
        pca,dur,*_=project.risk_models(history,jac,d)
        self.assertGreaterEqual(np.linalg.eigvalsh(pca).min(),-1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(dur).min(),-1e-12)

    def test_optimization_respects_equal_yield_and_cap(self):
        yields=np.array([.025,.03,.035,.04,.045,.05])
        w=project.optimize(np.diag([1,2,3,4,5,6]),yields,yields.mean())
        self.assertAlmostEqual(w.sum(),1,places=7)
        self.assertAlmostEqual(w@yields,yields.mean(),places=7)
        self.assertGreaterEqual(w.min(),-1e-8)
        self.assertLessEqual(w.max(),.4+1e-8)

    def test_future_changes_do_not_change_earlier_weights(self):
        dates=pd.bdate_range('2000-01-01','2001-06-01')
        t=np.arange(len(dates))
        data=pd.DataFrame(index=dates)
        for k,name in enumerate(project.PARAMETERS):
            data[name]=self.flat[k]
        data['BETA0']=4+.2*np.sin(t/12)
        data['BETA1']=.4*np.cos(t/19)
        data['BETA2']=.3*np.sin(t/25)
        curves=project.curve_panel(data)
        _,original,_,_=project.backtest(data,curves,'2001-01-01','2001-06-01',60)
        changed=data.copy()
        cutoff=pd.Timestamp('2001-04-01')
        changed.loc[changed.index>=cutoff,'BETA0']+=2
        _,other,_,_=project.backtest(changed,project.curve_panel(changed),'2001-01-01','2001-06-01',60)
        a=original.loc[original.date<cutoff].filter(like='weight_').to_numpy()
        b=other.loc[other.date<cutoff].filter(like='weight_').to_numpy()
        np.testing.assert_allclose(a,b,atol=1e-10)


if __name__=='__main__':
    unittest.main(verbosity=2)
