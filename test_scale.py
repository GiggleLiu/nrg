from numpy import *
from matplotlib.pyplot import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
import pdb,time

from scale import *
from nrgmap import get_ticker

class ScaleTest(object):
    def __init__(self):
        self.N=10
        self.Lambda=2.
        self.D=1.5
        self.ticker=get_ticker('log',Lambda=self.Lambda,D=self.D)
        tickers=[self.ticker,self.ticker]
        self.scale=ticker2scale(tickers,N=self.N,z=0.5)

    def test_saveload(self):
        token='test_ticker'
        self.scale.save(token)
        scale2=load_scale(token)
        assert_(all(scale2._data==self.scale._data))
        assert_(all(scale2._data_neg==self.scale._data_neg))
        assert_(scale2.z==self.scale.z)
        assert_(scale2.N==self.scale.N)

    def test_interval(self):
        assert_almost_equal(self.scale[-1],-self.ticker.D)
        assert_almost_equal(self.scale[1],self.ticker.D)
        assert_almost_equal(-self.scale.D[0],self.ticker.D)
        assert_almost_equal(self.scale.D[1],self.ticker.D)
        for i in xrange(self.N):
            z=self.scale.z
            assert_allclose(self.scale.interval(i+1),self.D*array([self.Lambda**-(i+z),self.Lambda**(0 if i==0 else (-i+1-z))]))
            assert_allclose(self.scale.interval(-i-1),-self.scale.interval(i+1)[::-1])

    def test_show(self):
        self.scale.show()
        show()

    def test_all(self):
        self.test_interval()
        self.test_saveload()
        self.test_show()

ScaleTest().test_all()
