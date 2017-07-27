import numpy as np
from src.core.transform import *

def test():
    p0 = Point(1,1,1)
    p1 = Point(2,3,5)
    v = Vector(1,2,3)
    a = Transform.look_at(p0, p1, v)
    b = Transform.rotate(36, v)
    assert feq(np.linalg.inv(a.mInv),a.m).all()
    assert feq(np.linalg.inv(b.mInv),b.m).all()
