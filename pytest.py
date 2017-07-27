from src.core.transform import *

p0 = Point(0,0,0)
p1 = Point(1,1,1)
v = Vector(1,1,1)

T = Transform.look_at(p0, p1, v)
T.m.dot(T.mInv)

