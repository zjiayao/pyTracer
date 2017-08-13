"""
triangle.py

Triangle and Triangle Mesh implementation.

Created by Jiayao on July 27, 2017
Modified on Aug 13, 2017
"""
from __future__ import absolute_import
from .. import *
from . import Shape
from .. import geometry as geo
from .. import transform as trans

__all__ = ['create_triangle_mesh', 'Triangle', 'TriangleMesh']

def create_triangle_mesh(o2w: 'trans.Transform', w2o: 'trans.Transform',
                         ro: bool, params: {str: object}, txt: {str: object} = None):
	"""Create triangle mesh from file"""

	vi = None if 'indices' not in params else params['indices']
	P = None if 'P' not in params else params['P']
	uvs = None if 'uv' not in params else params['uv']
	if uvs is None:
		uvs = None if 'st' not in params else params['st']
	discard = False if 'discard' not in params else params['discard']

	nvi = len(vi) if vi is not None else -1
	npi = len(P) if P is not None else -1
	nuvi = len(uvs) if uvs is not None else -1

	if uvs is not None:
		if nuvi < 2 * npi:
			raise RuntimeError('src.core.shape.create_triangle_mesh(): insufficient '
			                   '\"uv\" for triangular mesh, {} expected, {} found.' \
			                   .format(2 * npi, nuvi))
		elif nuvi > 2 * npi:
			print('[Warning] src.core.shape.create_triangle_mesh(): more \"uv\"s '
			      'found, {} expcted, {} found'.format(2 * npi, nuvi))

	if vi is None or P is None:
		return None

	S = None if 'S' not in params else params['S']
	nsi = len(S) if S is not None else -1
	if S is not None and nsi != npi:
		raise RuntimeError('src.core.shape.create_triangle_mesh(): \"S\" and \"P\" do not match')

	N = None if 'N' not in params else params['S']
	nni = len(N) if N is not None else -1
	if N is not None and nni != npi:
		raise RuntimeError('src.core.shape.create_triangle_mesh(): \"N\" and \"P\" do not match')

	if discard and uvs is not None and N is not None:
		# discard degenerated uvs if any
		for i in range(0, nvi, 3):
			area = .5 * (P[vi[i]] - P[vi[ i +1]]).cross(P[vi[ i +2]] - P[vi[ i +1]]).length()
			if area < EPS:
				continue
			if (uvs[2 * vi[i]] == uvs[2 * vi[ i +1]] and uvs[2 * vi[i] + 1] == uvs[2 * vi[ i +1] + 1]) or \
					(uvs[2 * vi[ i +1]] == uvs[2 * vi[ i +2]] and uvs[2 * vi[ i +1] + 1] == uvs[2 * vi[ i +2] + 1]) or \
					(uvs[2 * vi[ i +2]] == uvs[2 * vi[i]] and uvs[2 * vi[ i +2] + 1] == uvs[2 * vi[i] + 1]):
				print('[Warning] src.core.shape.create_triangle_mesh(): degenerated \"uv\"s, discarding')
				uvs = None
				nuvi = 0
				break
	for i in vi:
		if i >= npi:
			raise RuntimeError('src.core.shape.create_triangle_mesh(): mesh has out of-boundes vertex index')

	alphaTex = None


	if 'alphatex' in params:
		alphaTexStr = params['alphatex']
		if alphaTexStr in txt:
			alphaTex = txt[alphaTexStr]
		else:
			raise RuntimeError('src.core.shape.create_triangle_mesh(): counld not find texture {}' \
			                   .format(alphaTexStr))

	elif 'alpha' in params:
		pass
	# TODO
	# alphaTex = ConstantTexture(0.)

	return TriangleMesh(o2w, w2o, ro, nvi / 3, npi, vi, P, N, S, uvs, alphaTex)

class Triangle(Shape):
	"""
	Triangle Class

	Subclasses `Shape` and inner classes
	`TriangleMesh`. Holds refernces to
	the data in the outer `TriangleMesh`
	"""

	def __init__(self, o2w: 'trans.Transform', w2o: 'trans.Transform',
	             ro: bool, m: 'TriangleMesh', n: INT):
		self.mesh = m
		self.v = m.vertexIndex[3 * n]  # pbrt uses a pointer

	def __repr__(self):
		return "{}\nMesh: {}\nVertex Index: {}" \
			.format(self.__class__, self.mesh, self.v)

	def get_uvs(self) -> [FLOAT]:
		if self.mesh.uvs is None:
			return [[0., 0.], [1., 0.], [1., 1.]]  # default
		else:
			return [[self.mesh.uvs[2 * self.v], self.mesh.uvs[2 * self.v + 1]],
			        [self.mesh.uvs[2 * (self.v + 1)], self.mesh.uvs[2 * (self.v + 1) + 1]],
			        [self.mesh.uvs[2 * (self.v + 2)], self.mesh.uvs[2 * (self.v + 2) + 1]]]

	def object_bound(self) -> 'geo.BBox':
		p1 = self.mesh.p[self.v]
		p2 = self.mesh.p[self.v + 1]
		p3 = self.mesh.p[self.v + 2]
		return geo.BBox(self.mesh.w2o(p1), self.mesh.w2o(p2)) \
			.union(self.mesh.w2o(p3))

	def world_bound(self) -> 'geo.BBox':
		p1 = self.mesh.p[self.v]
		p2 = self.mesh.p[self.v + 1]
		p3 = self.mesh.p[self.v + 2]
		return geo.BBox(p1, p2).union(p3)

	def sample(self, u1: FLOAT, u2: FLOAT) -> ['geo.Point', 'geo.Normal']:
		from ..montecarlo import uniform_sample_triangle
		b1, b2 = uniform_sample_triangle(u1, u2)

		p = b1 * self.mesh.p[self.v] + \
		    b2 * self.mesh.p[self.v + 1] + \
		    (1. - b1 - b2) * self.mesh.p[self.v + 2]

		Ns = geo.normalize(geo.Normal.fromVector(
			(self.mesh.p[self.v + 1] - self.mesh.p[self.v]) \
				.cross(self.mesh.p[self.v + 2] - self.mesh.p[self.v])))

		if self.ro:
			Ns *= -1.

		return [p, Ns]

	def intersect(self, r: 'Ray') -> (bool, FLOAT, FLOAT, 'geo.DifferentialGeometry'):
		"""
		Determine whether intersects
		using Barycentric coordinates
		"""
		# compute s1
		p1 = self.mesh.p[self.v]
		p2 = self.mesh.p[self.v + 1]
		p3 = self.mesh.p[self.v + 2]

		e1 = p2 - p1  # geo.Vector
		e2 = p3 - p1
		s1 = r.d.cross(e2)
		div = s1.dot(e1)

		if div == 0.:
			return (False, None, None, None)
		divInv = 1. / div

		# compute barycentric coordinate
		## first one
		d = r.o - p1
		b1 = d.dot(s1) * divInv
		if b1 < 0. or b1 > 1.:
			return (False, None, None, None)
		## second one
		s2 = d.cross(e1)
		b2 = r.d.dot(s2) * divInv
		if b2 < 0. or (b1 + b2) > 1.:
			return (False, None, None, None)

		# compute intersection
		t = e2.dot(s2) * divInv
		if t < r.mint or d > r.maxt:
			return (False, None, None, None)

		# compute partial derivatives
		uvs = self.get_uvs()
		du1 = uvs[0][0] - uvs[2][0]
		du2 = uvs[1][0] - uvs[2][0]
		dv1 = uvs[0][1] - uvs[2][1]
		dv2 = uvs[1][1] - uvs[2][1]
		dp1 = p1 - p3
		dp2 = p2 - p3

		det = du1 * dv2 - du2 * dv1
		if det == 0.:
			# choose an arbitrary system
			_, dpdu, dpdv = geo.coordinate_system(geo.normalize(e2.cross(e1)))
		else:
			detInv = 1. / det
			dpdu = (dv2 * dp1 - dv1 * dp2) * detInv
			dpdv = (-du2 * dp1 + du1 * dp2) * detInv

		# interpolate triangle parametric coord.
		b0 = 1. - b1 - b2
		tu = b0 * uvs[0][0] + b1 * uvs[1][0] + b2 * uvs[2][0]
		tv = b0 * uvs[0][1] + b1 * uvs[1][1] + b2 * uvs[2][1]

		# test alpha texture
		dg = geo.geo.DifferentialGeometry(r(t), dpdu, dpdv,
		                          geo.Normal(0., 0., 0.), geo.Normal(0., 0., 0.),
		                          tu, tv, self)

		if self.mesh.alphaTexture is not None:
			# alpha mask presents
			if self.mesh.alphaTexture.evaluate(dg) == 0.:
				return (False, None, None, None)

		# have a hit
		return True, t, 1e-3 * t, dg

	def intersectP(self, r: 'Ray') -> bool:
		"""
		Determine whether intersects
		using Barycentric coordinates
		"""
		# compute s1
		p1 = self.mesh.p[self.v]
		p2 = self.mesh.p[self.v + 1]
		p3 = self.mesh.p[self.v + 2]

		e1 = p2 - p1  # geo.Vector
		e2 = p3 - p1
		s1 = r.d.cross(e2)
		div = s1.dot(e1)

		if div == 0.:
			return False
		divInv = 1. / div

		# compute barycentric coordinate
		## first one
		d = r.o - p1
		b1 = d.dot(s1) * divInv
		if b1 < 0. or b1 > 1.:
			return False
		## second one
		s2 = d.cross(e1)
		b2 = r.d.dot(s2) * divInv
		if b2 < 0. or (b1 + b2) > 1.:
			return False

		# compute intersection
		t = e2.dot(s2) * divInv
		if t < r.mint or d > r.maxt:
			return False

		# compute partial derivatives
		uvs = self.get_uvs()
		du1 = uvs[0][0] - uvs[2][0]
		du2 = uvs[1][0] - uvs[2][0]
		dv1 = uvs[0][1] - uvs[2][1]
		dv2 = uvs[1][1] - uvs[2][1]
		dp1 = p1 - p3
		dp2 = p2 - p3

		det = du1 * dv2 - du2 * dv1
		if det == 0.:
			# choose an arbitrary system
			_, dpdu, dpdv = geo.coordinate_system(geo.normalize(e2.cross(e1)))
		else:
			detInv = 1. / det
			dpdu = (dv2 * dp1 - dv1 * dp2) * detInv
			dpdv = (-du2 * dp1 + du1 * dp2) * detInv

		# interpolate triangle parametric coord.
		b0 = 1. - b1 - b2
		tu = b0 * uvs[0][0] + b1 * uvs[1][0] + b2 * uvs[2][0]
		tv = b0 * uvs[0][1] + b1 * uvs[1][1] + b2 * uvs[2][1]

		# test alpha texture
		dg = geo.geo.DifferentialGeometry(r(t), dpdu, dpdv,
		                          geo.Normal(0., 0., 0.), geo.Normal(0., 0., 0.),
		                          tu, tv, self)

		if self.mesh.alphaTexture is not None:
			# alpha mask presents
			if self.mesh.alphaTexture.evaluate(dg) == 0.:
				return False

		# have a hit
		return True

	def area(self) -> FLOAT:
		p1 = self.mesh.p[self.v]
		p2 = self.mesh.p[self.v + 1]
		p3 = self.mesh.p[self.v + 2]
		return .5 * (p2 - p1).cross(p3 - p1).length()

	def get_shading_geometry(self, o2w: 'trans.Transform',
	                         dg: 'geo.DifferentialGeometry') -> 'geo.DifferentialGeometry':
		if self.mesh.n is not None and self.mesh.s is not None:
			return dg
		# compute barycentric coord
		uvs = self.get_uvs()
		A = np.array([[uvs[1][0] - uvs[0][0], uvs[2][0] - uvs[0][0]],
		              [uvs[1][1] - uvs[0][1], uvs[2][1] - uvs[0][1]]],
		             dtype=FLOAT)
		C = np.array([dg.u - uvs[0][0], dg.v - uvs[0][1]])
		try:
			b = np.linalg.solve(A, C)
		except:
			b = [1. / 3, 1. / 3, 1. / 3]

		else:
			b = [1. - b[1] - b[2], b[1], b[2]]

		# compute shading tangents
		if self.mesh.n is not None:
			ns = geo.normalize(o2w(b[0] * self.mesh.n[self.v] +
			                   b[1] * self.mesh.n[self.v + 1] +
			                   b[2] * self.mesh.n[self.v + 2]))
		else:
			ns = dg.nn

		if self.mesh.s is not None:
			ss = geo.normalize(o2w(b[0] * self.mesh.s[self.v] +
			                   b[1] * self.mesh.s[self.v + 1] +
			                   b[2] * self.mesh.s[self.v + 2]))
		else:
			ss = geo.normalize(dg.dpdu)

		ts = ss.cross(ns)
		if ts.sq_length() > 0.:
			ts.normalize()
			ss = ts.cross(ns)
		else:
			_, ss, ts = geo.coordinate_system(ns)

		# compute dndu and dndv
		if self.mesh.n is not None:
			du1 = uvs[0][0] - uvs[2][0]
			du2 = uvs[1][0] - uvs[2][0]
			dv1 = uvs[0][1] - uvs[2][1]
			dv2 = uvs[1][1] - uvs[2][1]
			dn1 = self.mesh.n[self.v] - self.mesh.n[self.v + 2]
			dn2 = self.mesh.n[self.v + 1] - self.mesh.n[self.v + 2]

			det = du1 * dv2 - du2 * dv1
			if det == 0.:
				# choose an arbitrary system
				dndu = dndv = geo.Normal(0., 0., 0.)
			else:
				detInv = 1. / det
				dndu = (dv2 * dn1 - dv1 * dn2) * detInv
				dndv = (-du2 * dn1 + du1 * dn2) * detInv

		dg = geo.geo.DifferentialGeometry(dg.p, ss, ts,
		                          self.mesh.o2w(dndu), self.mesh.o2w(dndv), dg.u, dg.v, dg.shape)
		# todo
		return dg


class TriangleMesh(Shape):
	"""
	TriangleMesh Class

	Subclasses `Shape` and is used
	to model trianglular meshes.
	"""

	def __init__(self, o2w: 'trans.Transform', w2o: 'trans.Transform',
	             ro: bool, nt: INT, nv: INT, vi: [INT],
	             P: [geo.Point], N: [geo.Normal] = None, S: [geo.Vector] = None,
	             uv: np.ndarray = None, atex=None):
		"""
		o2w, w2o: Transformations
		ro: reverse_orientation
		nt: # of triangles
		nv: # of vertices
		vi: plain array of vertex indices
		P: plain array of `geo.Point`s
			i-th triangle: P[vi[3*i]], P[vi[3*i+1]], P[vi[3*i+2]]
		N: plain array of `geo.Normal`s
		S: plain array of `geo.Vector`s
		uv: plain array of parametric values
		atex: reference to alpha mask texture
		"""
		super().__init__(o2w, w2o, ro)
		self.alphaTexture = atex
		self.ntris = nt
		self.nverts = nv
		self.vertexIndex = vi.copy()
		self.uvs = None if uv is None else uv.copy()
		self.n = None if N is None else N.copy()
		self.s = None if S is None else S.copy()
		# transform the mesh to the world system
		self.p = [o2w(p) for p in P]

	def __repr__(self):
		return "{}\nTriangles: {}\nVertices: {}" \
			.format(self.__class__, self.ntris, self.nverts)

	# assumes the caller will cache the result
	def object_bound(self) -> 'geo.BBox':
		ret = geo.BBox()
		for i in range(self.nverts):
			ret = ret.union(self.w2o(self.p[i]))
		return ret

	def world_bound(self) -> 'geo.BBox':
		ret = geo.BBox()
		for i in range(self.nverts):
			ret = ret.union(self.p[i])
		return ret

	def can_intersect(self) -> bool:  # why didn't pbrt make it an attribute?
		return False

	# produce a list of `Shape`s that
	# can be intersected
	def refine(self) -> ['Shape']:
		"""
		returns a list of triangle references
		"""
		return [Triangle(self.o2w, self.w2o, self.ro, self, i) for i in range(self.ntris)]

