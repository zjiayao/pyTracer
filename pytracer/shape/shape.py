"""
shape.py

The base class of shapes.
All `Shape`s is desgined in the
object coordinate system.

Shapes include:
	- LoopSubdiv
	- TriangleMesh
	- Sphere
	- Cylinder
	- Disk

Created by Jiayao on July 27, 2017
"""
from src.montecarlo.montecarlo import *
# import numpy as np
# from abc import ABCMeta, abstractmethod
# from src.core.pytracer import *
# from src.core.geometry import *
# from src.core.transform import *


# aux functions for mesh navigation
def NEXT(i: INT) -> INT:
	return (i + 1) % 3
def PREV(i: INT) -> INT:
	return (i + 2) % 3

def create_triangle_mesh(o2w: 'Transform', w2o: 'Transform',
			ro: bool, params: {str: object}, txt: {str: object} = None):
	
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
			area = .5 * (P[vi[i]] - P[vi[i+1]]).cross(P[vi[i+2]] - P[vi[i+1]]).length()
			if area < EPS:
				continue
			if (uvs[2 * vi[i]] == uvs[2 * vi[i+1]] and uvs[2 * vi[i] + 1] == uvs[2 * vi[i+1] + 1]) or \
			   (uvs[2 * vi[i+1]] == uvs[2 * vi[i+2]] and uvs[2 * vi[i+1] + 1] == uvs[2 * vi[i+2] + 1]) or\
			   (uvs[2 * vi[i+2]] == uvs[2 * vi[i]] and uvs[2 * vi[i+2] + 1] == uvs[2 * vi[i] + 1]):
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

def create_loop_subdiv(o2w: 'Transform', w2o: 'Transform',
			ro: bool, params: {str: object}):
	nlevels = 3 if 'nlevels' not in params else params['nlevels']
	vi = None if 'indices' not in params else params['indices']
	P = None if 'P' not in params else params['p']

	if vi is None or P is None:
		return None

	return LoopSubdiv(o2w, w2o, ro, len(vi) / 3, len(P), vi, P, nlevels)

	



class LoopSubdiv(Shape):
	"""
	LoopSubdiv Class

	Implement Loop's method for subdivision surfaces.
	Assumes meshes are manifolds and are consistently ordered.
	"""
	class SDVertex():
		"""
		SDVertex Class

		Inner class for `LoopSubdiv`
		"""
		def __init__(self, pt: Point = Point(0., 0., 0.,)):
			self.P = pt.copy()
			self.startFace = None # ref to SDFace
			self.child = None # ref to SDVertex
			self.regular = self.boundary = False

		def __repr__(self):
			return "{}\nPoint: {}".format(self.__class__, self.P)

		def __lt__(self, other):
			return id(self) < id(other)

		def __eq__(self, other):
			return id(self) == id(other)

		def __hash__(self):
			return id(self)

		def valence(self):
			f = self.startFace
			if self.boundary == False:
				# interior
				nf = 1
				f = f.next_face(self)
				while f != self.startFace:
					nf += 1
					f = f.next_face(self)
				return nf 
			else:
				# boudary
				nf = 1
				f = f.next_face(self)
				while f is not None:
					nf += 1
					f = f.next_face(self)
				f = self.startFace(self).prev_face(self)
				while f is not None:
					nf += 1
					f = f.prev_face(self)
				return nf + 1

		def one_ring(self):
			P = []
			if not self.boundary:
				# interior
				f = self.startFace
				P.append(f.next_vert(self).P)
				f = f.next_face(self)
				while f != self.startFace:
					P.append(f.next_vert(self).P)
					f = f.next_face(self)
			else:
				# boundary
				f = self.startFace
				f2 = f.next_face(self)
				while f2 is not None:
					f = f2
					f2 = f.next_face(self)

				P.append(f.next_vert(self).P)
				P.append(f.prev_vert(self).P)
				f = f.prev_face(self)
				while f is not None:
					P.append(f.prev_vert(self).P)
					f = f.prev_face(self)

			return P


	class SDFace():
		"""
		SDFace Class

		Inner class for `LoopSubdiv`
		"""
		def __init__(self):
			self.v = [None, None, None] # ref to SDVertex (3)
			self.f = [None, None, None] # ref to SDFace (3)
			self.children = [None, None, None, None] # ref to SDFace (4)

		def __repr__(self):
			return "{}".format(self.__class__)

		def vnum(self, vert) -> INT:
			for i in range(3):
				if self.v[i] == vert:
					return i
			raise RuntimeError('Logic error in Shape.LoopSubdiv.SDFace.vnum()')

		def next_face(self, vert):
			return self.f[self.vnum(vert)]

		def prev_face(self, vert):
			return self.f[PREV(self.vnum(vert))]

		def next_vert(self, vert):
			return self.v[self.vnum(vert)]

		def prev_vert(self, vert):
			return self.v[PREV(self.vnum(vert))]

		def other_vert(self, v0, v1):
			for i in range(3):
				if self.v[i] != v0 and self.v[i] != v1:
					return self.v[i]
			raise RuntimeError('Logic error in Shape.LoopSubdiv.SDVertex.other_vert()')

	class SDEdge():
		"""
		SDEdge Class

		Inner class for `LoopSubdiv`
		"""
		def __init__(self, v0: 'SDVertex' = None, v1: 'SDVertex' = None):
			self.v = [v0, v1] if id(v0) < id(v1) else [v1, v0]
			self.f = [None, None]
			self.f0edge_num = -1

		def __repr__(self):
			return "{}".format(self.__class__)

		def __lt__(self, other):
			if id(self.v[0]) == id(other.v[0]):
				return id(self.v[1]) < id(other.v[1])
			return id(self.v[0]) < id(other.v[0])

		def __eq__(self, other):
			return id(self.v[0]) == id(other.v[0]) and id(self.v[1]) == id(other.v[1])

		def __hash__(self):
			return int(str(id(self.v[0]) + id(self.v[1])))

		

	def __init__(self, o2w: 'Transform', w2o :'Transform',
				ro: bool, nf: INT, nv: INT, vi: [INT],
				P: [Point], nl: INT):
		super().__init__(o2w, w2o, ro)
		self.nLevels = nl
		self.vertices = np.array([self.SDVertex(P[i]) for i in range(nv)])
		self.faces = np.array([self.SDFace() for i in range(nf)])

		# set face to vertex refs
		for i in range(nf):
			f = self.faces[i]
			k = 0
			f.v = [self.vertices[vi[k+j]] for j in range(3)]			
			for j in range(3):
				f.v[j].startFace = f
			k += 3

		# set neighbor refs in faces
		edges = {}
		for i in range(nf):
			f = self.faces[i]
			for edge_num in range(3):
				v0 = edge_num
				v1 = NEXT(edge_num)
				e = LoopSubdiv.SDEdge(f.v[v0], f.v[v1])
				if e in edges:
					e = edges[e]
					e.f[0].f[e.f0edge_num] = f
					f.f[edge_num] = e.f[0]
					del edges[e] # by assumption, each edge appears at most twice
				else:
					e.f[0] = f
					e.f0edge_num = edge_num
					edges[e] = e

		# finish vertex init
		for i in range(nv):
			v = self.vertices[i]
			f = v.startFace
			f = f.next_face(v)
			while f is not None and not f == v.startFace:
				f = f.next_face(v)
			v.boundary = (f == None)				

			if v.boundary is not None and v.valence() == 6:
				v.regular = True
			elif v.boundary is not None and v.valence() == 4:
				v.regular = True
			else:
				v.regular = False

	def __repr__(self):
		return "{}\nLevels: {}".format(self.__class__, self.nLevels)

	@staticmethod
	def beta(valence: INT) -> FLOAT:
		if valence == 3:
			return .1875
		else:
			return 3. / (8. * valence)

	@staticmethod
	def gamma(valence: INT) -> FLOAT:
		return 1. / (valence + 3. / (8. + LoopSubdiv.beta(valence)))

	def weight_one_ring(self, vert: 'SDVertex', beta: 'FLOAT') -> 'Point':
		valence = vert.valence()
		Pring = vert.one_ring()
		P = (1. - valence * beta) * vert.P
		for p in Pring:
			P += beta * p
		return P

	def weight_boundary(self, vert: 'SDVertex', beta: 'FLOAT') -> 'Point':
		valence = vert.valence()
		Pring = vert.one_ring()
		P = (1. - 2. * beta) * vert.P
		P += beta * (Pring[0] + Pring[-1])
		return P

	def object_bound(self) -> 'BBox':
		ret = BBox()
		for i in range(len(self.vertices)):
			ret.union(self.vertices[i].P)
		return ret

	def world_bound(self) -> 'BBox':
		ret = BBox()
		for i in range(len(self.vertices)):
			ret.union(self.o2w(self.vertices[i].P))
		return ret

	def can_intersect(self) -> bool:
		return False

	def refine(self) -> ['Shape']:
		f = self.faces
		v = self.vertices
		for i in range(self.nLevels):
			newFaces = []
			newVertices = []
			# update next level
			for vv in v:
				vv.child = LoopSubdiv.SDVertex()
				vv.child.regular = vv.regular
				vv.child.boundary = vv.boundary
				newVertices.append(vv.child)

			for ff in f:
				for k in range(4):
					ff.children[k] = LoopSubdiv.SDFace()
					newFaces.append(ff.children[k])

			# update vertex positions and create new edge vertices
			## update for even vertices (already presented)
			for vv in v:
				if not vv.boundary:
					# one-ring rule
					# $v' = (1 - n \beta) v + \sum_{i=1}^{N} \beta v_i$
					if vv.regular:
						vv.child.P = self.weight_one_ring(vv, .0625)
					else:
						vv.child.P = self.weight_one_ring(vv, LoopSubdiv.beta(vv.valence()))
				else:
					# boundary rule
					# $v' = (1 - 2 \beta) v + \beta (v_1 + v_2)
					vv.child.P = self.weight_boundary(vv, .125)

			## update for odd vertices (along the split edges)
			edgeVerts = {}
			for ff in f:
				for k in range(3):
					# k-th edge
					edge = LoopSubdiv.SDEdge(ff.v[k], ff.v[NEXT(k)])

					if edge not in edgeVerts:
						# create and init
						vert = LoopSubdiv.SDVertex()
						newVertices.append(vert)
						vert.regular = True
						vert.boundary = (ff.f[k] is None)
						vert.startFace = ff.children[3]

						# compute new vertex position
						if vert.boundary:
							vert.P = .5 * edge.v[0].P + .5 * edge.v[1].P
						else:
							vert.P = .375 * edge.v[0].P + \
									 .375 * edge.v[1].P + \
									 .125 * ff.other_vert(edge.v[0], edge.v[1]).P + \
									 .125 * ff.f[k].other_vert(edge.v[0], edge.v[1]).P
						edgeVerts[edge] = vert


			# update new mesh topology
			## startFace ref of odd vertices (done)
			## startFace ref of even vertices
			for vv in v:
				v_num = vv.startFace.vnum(v)
				v.child.startFace = vv.startFace.child[v_num]

			## new faces' neighbor ref
			for ff in f:
				for k in range(3):
					### update for siblings
					ff.children[3].f[k] = ff.children[NEXT(k)]
					ff.children[k].f[NEXT(k)] = ff.children[3]

					### update for neighbor children
					f2 = ff.f[k]
					ff.children[k].f[k] = None if f2 is None else f2.children[f2.vnum(ff.v[k])]
					f2 = ff.f[PREV(k)]
					ff.children[k].f[PREV(k)] = None if f2 is None else f2.children[f2.vnum(ff.v[k])]

			## new faces' vertices ref
			for ff in f:
				for k in range(3):
					### update to new even vertex
					ff.children[k].v[k] = ff.v[k].child
					### update to new odd vertex
					vert = edgeVerts[ LoopSubdiv.SDEdge(ff.v[k], ff.v[NEXT(k)]) ]
					ff.children[k].v[NEXT(k)] = vert
					ff.children[NEXT(k)].v[k] = vert
					ff.children[3].v[k] = vert

			# prepare for next level
			f = np.array(newFaces)
			v = np.array(newVertices)

		# add vertices to the limiting surface
		Plimit = []# [Point]
		for vv in v:
			if vv.boundary:
				Plimit.append( self.weight_boundary(vv, .2) )
			else:
				Plimit.append( self.weight_one_ring(vv, self.gamma(vv.valence())) )

		for i, vv in enumerate(v):
			vv.p = Plimit[i]


		# compute vertex tangents
		## S: first tangent, across tangent
		## T: second tangent, transverse tangent
		Ns = []
		for vv in v:
			valence = vv.valence()
			valInv = 1. / valence

			Pring = vv.one_ring()

			if not vv.boundary:
				## interior
				S += np.sum([np.cos(2. * np.pi * k * valInv) * (Pring[k]) for k in range(valence)], axis=0)
				T += np.sum([np.sin(2. * np.pi * k * valInv) * (Pring[k]) for k in range(valence)], axis=0)


			else:
				## boundary
				S = Pring[valence-1] - Pring[0]
				if valence == 2:
					T = Vector.fromPoint(Pring[0] + Pring[1] -2 * vert.P)
				elif valence == 3:
					T = Pring[1] - v.P
				elif valence == 4:
					T = Vector(-Pring[0] + 2 * Pring[1] + 2 * Pring[2] + -Pring[3] + -2 * vv.P) # avoid Point substraction
				else:
					theta = np.pi / (valence - 1)
					T = Vector( np.sin(theta) * (Pring[0] + Pring[-1]) )
					T += np.sum([( (2 * np.cos(theta) - 2) * np.sin(k * theta) ) * \
										 Pring[k] for k in range(valence-1)], axis=0)
					T = -T

			Ns.append(Normal.fromVector(S.cross(T)))


		# create `TriangleMesh`
		verts = []
		usedVerts = {}
		for i, vv in enumerate(v):
			usedVerts[vv] = i
		for ff in f:
			for j in range(3):
				verts.append(usedVerts[ff.v[j]])
		
		params = {	'indices': verts,
					'P': Plimit,
					'N': Ns }
		return [create_triangle_mesh(self.o2w, self.w2o, self.ro, params)]







class Triangle(Shape):
	"""
	Triangle Class

	Subclasses `Shape` and inner classes
	`TriangleMesh`. Holds refernces to
	the data in the outer `TriangleMesh`
	"""
	def __init__(self, o2w: 'Transform', w2o: 'Transform',
			ro: bool, m: 'TriangleMesh', n: INT):
		self.mesh = m
		self.v = m.vertexIndex[3*n] # pbrt uses a pointer

	def __repr__(self):
		return "{}\nMesh: {}\nVertex Index: {}" \
			.format(self.__class__, self.mesh, self.v)

	def get_uvs(self) -> [FLOAT]:
		if self.mesh.uvs is None:
			return [[0., 0.], [1., 0.], [1., 1.]] # default
		else:
			return [[self.mesh.uvs[2 * self.v], self.mesh.uvs[2 * self.v + 1]],
					[self.mesh.uvs[2 * (self.v+1)], self.mesh.uvs[2 * (self.v+1) + 1]],
					[self.mesh.uvs[2 * (self.v+2)], self.mesh.uvs[2 * (self.v+2) + 1]]]


	def object_bound(self) -> 'BBox':
		p1 = self.mesh.p[self.v]
		p2 = self.mesh.p[self.v+1]
		p3 = self.mesh.p[self.v+2]
		return BBox(self.mesh.w2o(p1), self.mesh.w2o(p2)) \
				.union(self.mesh.w2o(p3))
	
	def world_bound(self) -> 'BBox':
		p1 = self.mesh.p[self.v]
		p2 = self.mesh.p[self.v+1]
		p3 = self.mesh.p[self.v+2]			
		return BBox(p1, p2).union(p3)

	def sample(self, u1: FLOAT, u2: FLOAT) -> ['Point', 'Normal']:
		b1, b2 = uniform_sample_triangle(u1, u2)

		p = b1 * self.mesh.p[self.v] + \
			b2 * self.mesh.p[self.v+1] + \
			(1. - b1 - b2) * self.mesh.p[self.v+2]	

		Ns = normalize(Normal.fromVector(
			(self.mesh.p[self.v+1] - self.mesh.p[self.v]) \
				.cross(self.mesh.p[self.v+2]-self.mesh.p[self.v])))

		if self.ro:
			Ns *= -1.

		return [p, Ns]

	def intersect(self, r: 'Ray') -> (bool, FLOAT, FLOAT, 'DifferentialGeometry'):
		"""
		Determine whether intersects
		using Barycentric coordinates
		"""
		# compute s1
		p1 = self.mesh.p[self.v]
		p2 = self.mesh.p[self.v+1]
		p3 = self.mesh.p[self.v+2]	

		e1 = p2 - p1 # Vector
		e2 = p3 - p1
		s1 = r.d.cross(e2)
		div =  s1.dot(e1)

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
			_, dpdu, dpdv = coordinate_system(normalize(e2.cross(e1)))
		else:
			detInv = 1. / det
			dpdu = (dv2 * dp1 - dv1 * dp2) * detInv
			dpdv = (-du2 * dp1 + du1 * dp2) * detInv

		# interpolate triangle parametric coord.
		b0 = 1. - b1 - b2
		tu = b0 * uvs[0][0] + b1 * uvs[1][0] + b2 * uvs[2][0]
		tv = b0 * uvs[0][1] + b1 * uvs[1][1] + b2 * uvs[2][1]

		# test alpha texture
		dg = DifferentialGeometry(r(t), dpdu, dpdv,
				Normal(0., 0., 0.), Normal(0., 0., 0.),
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
		p2 = self.mesh.p[self.v+1]
		p3 = self.mesh.p[self.v+2]	

		e1 = p2 - p1 # Vector
		e2 = p3 - p1
		s1 = r.d.cross(e2)
		div =  s1.dot(e1)

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
			_, dpdu, dpdv = coordinate_system(normalize(e2.cross(e1)))
		else:
			detInv = 1. / det
			dpdu = (dv2 * dp1 - dv1 * dp2) * detInv
			dpdv = (-du2 * dp1 + du1 * dp2) * detInv

		# interpolate triangle parametric coord.
		b0 = 1. - b1 - b2
		tu = b0 * uvs[0][0] + b1 * uvs[1][0] + b2 * uvs[2][0]
		tv = b0 * uvs[0][1] + b1 * uvs[1][1] + b2 * uvs[2][1]

		# test alpha texture
		dg = DifferentialGeometry(r(t), dpdu, dpdv,
				Normal(0., 0., 0.), Normal(0., 0., 0.),
				tu, tv, self)

		if self.mesh.alphaTexture is not None:
			# alpha mask presents
			if self.mesh.alphaTexture.evaluate(dg) == 0.:
				return False

		# have a hit
		return True		
	
	def area(self) -> FLOAT:
		p1 = self.mesh.p[self.v]
		p2 = self.mesh.p[self.v+1]
		p3 = self.mesh.p[self.v+2]	
		return .5 * (p2-p1).cross(p3-p1).length()

	def get_shading_geometry(self, o2w: 'Transform',
			dg: 'DifferentialGeometry') -> 'DifferentialGeometry':
		if self.mesh.n is not None and self.mesh.s is not None:
			return dg
		# compute barycentric coord
		uvs = self.get_uvs()
		A = np.array([[ uvs[1][0] - uvs[0][0], uvs[2][0] - uvs[0][0] ],
					  [ uvs[1][1] - uvs[0][1], uvs[2][1] - uvs[0][1] ]],
					  dtype=FLOAT)
		C = np.array([dg.u - uvs[0][0], dg.v - uvs[0][1]])
		try:
			b = np.linalg.solve(A, C)
		except:
			b = [1./3, 1./3, 1./3]

		else:
			b = [1. - b[1] - b[2], b[1], b[2]]


		# compute shading tangents
		if self.mesh.n is not None:
			ns = normalize(o2w(b[0] * self.mesh.n[self.v] +
							   b[1] * self.mesh.n[self.v+1] +
							   b[2] * self.mesh.n[self.v+2] ))
		else:
			ns = dg.nn

		if self.mesh.s is not None:
			ss = normalize(o2w(b[0] * self.mesh.s[self.v] +
							   b[1] * self.mesh.s[self.v+1] +
							   b[2] * self.mesh.s[self.v+2] ))
		else:
			ss = normalize(dg.dpdu)

		ts = ss.cross(ns)
		if ts.sq_length() > 0.:
			ts.normalize()
			ss = ts.cross(ns)
		else:
			_, ss, ts = coordinate_system(ns)
		
		# compute dndu and dndv
		if self.mesh.n is not None:
			du1 = uvs[0][0] - uvs[2][0]
			du2 = uvs[1][0] - uvs[2][0]
			dv1 = uvs[0][1] - uvs[2][1]
			dv2 = uvs[1][1] - uvs[2][1]
			dn1 = self.mesh.n[self.v] - self.mesh.n[self.v+2]
			dn2 = self.mesh.n[self.v+1] - self.mesh.n[self.v+2]

			det = du1 * dv2 - du2 * dv1
			if det == 0.:
				# choose an arbitrary system
				dndu = dndv = Normal(0., 0., 0.)
			else:
				detInv = 1. / det
				dndu = (dv2 * dn1 - dv1 * dn2) * detInv
				dndv = (-du2 * dn1 + du1 * dn2) * detInv			

		dg = DifferentialGeometry(dg.p, ss, ts,
			self.mesh.o2w(dndu), self.mesh.o2w(dndv), dg.u, dg.v, dg.shape)
		# todo
		return dg

class TriangleMesh(Shape):
	"""
	TriangleMesh Class

	Subclasses `Shape` and is used
	to model trianglular meshes.
	"""	
	def __init__(self, o2w: 'Transform', w2o :'Transform',
			ro: bool, nt: INT, nv: INT, vi: [INT],
			P: [Point], N: [Normal] = None, S: [Vector] = None,
			uv: np.ndarray = None, atex = None):
		"""
		o2w, w2o: Transformations
		ro: reverse_orientation
		nt: # of triangles
		nv: # of vertices
		vi: plain array of vertex indices
		P: plain array of `Point`s
			i-th triangle: P[vi[3*i]], P[vi[3*i+1]], P[vi[3*i+2]]
		N: plain array of `Normal`s
		S: plain array of `Vector`s
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
	def object_bound(self) -> 'BBox':
		ret = BBox()
		for i in range(self.nverts):
			ret = ret.union(self.w2o(self.p[i]))
		return ret

	def world_bound(self) -> 'BBox':
		ret = BBox()
		for i in range(self.nverts):
			ret = ret.union(self.p[i])
		return ret

	def can_intersect(self) -> bool:	# why didn't pbrt make it an attribute?
		return False

	# produce a list of `Shape`s that
	# can be intersected
	def refine(self) -> ['Shape']:
		"""
		returns a list of triangle references
		"""
		return [Triangle(self.o2w, self.w2o, self.ro, self, i) for i in range(self.ntris)]
	
class Sphere(Shape):
	"""
	Sphere Class

	Subclasses `Shape` and is used
	to model possibly partial Sphere.
	"""
	def __init__(self, o2w: 'Transform', w2o :'Transform',
			ro: bool, rad: FLOAT, z0: FLOAT, z1: FLOAT, pm: FLOAT):
		super().__init__(o2w, w2o, ro)
		self.radius = rad
		self.zmin = np.clip(min(z0, z1), -rad, rad)
		self.zmax = np.clip(max(z0, z1), -rad, rad)
		self.thetaMin = np.arccos(np.clip(self.zmin / rad, -1., 1.))
		self.thetaMax = np.arccos(np.clip(self.zmax / rad, -1., 1.))
		self.phiMax = np.deg2rad(np.clip(pm, 0., 360.))

	def __repr__(self):
		return "{}\nRadius: {}".format(self.__class__, self.radius)

	def object_bound(self) -> 'BBox':
		return BBox(Point(-self.radius, -self.radius, self.zmin),
					Point(self.radius, self.radius, self.zmax))
	@jit
	def sample(self, u1: FLOAT, u2: FLOAT) -> ['Point', 'Normal']:
		"""
		account for partial sphere
		"""
		v = uniform_sample_sphere(u1, u2)	
		phi = spherical_theta(v) * self.phiMax * INV_2PI	
		theta = self.thetaMin + spherical_theta(v) * (self.thetaMax - self.thetaMin)
		
		v = spherical_direction(np.sin(theta), np.cos(theta), phi) * self.radius
		v.z = self.zmin + v.z * (self.zmax - self.zmin)	

		p = Point.fromVector(v)
		Ns = normalize(self.o2w(Normal(p.x, p.y, p.z)))
		if self.ro:
			Ns *= -1.
		return [self.o2w(p), Ns]			

		# """
		# Not account for partial sphere
		# """
		# p = Point.fromVector(radius * uniform_sample_sphere(u1, u2))
		# Ns = normalize(self.o2w(Normal(p.x, p.y, p.z)))
		# if self.ro:
		# 	Ns *= -1.
		# return [self.o2w(p), Ns]
		
	def refine(self) -> ['Shape']:
		"""
		If `Shape` cannot intersect,
		return a refined subset
		"""
		raise NotImplementedError('Intersecable shapes are not refineable')

	def sample_p(self, pnt: 'Point', u1: FLOAT, u2: FLOAT) -> ['Point', 'Normal']:
		"""
		uniformly sample the sphere
		visible (of certain solid angle)
		to the point
		"""
		# compute coords for sampling
		ctr = self.o2w(Point(0., 0., 0.))
		wc = normalize(ctr - pnt)
		_, wc_x, wc_y = coordinate_system(wc)

		# sample uniformly if p is inside
		if pnt.sq_dist(ctr) - self.radius * self.radius < EPS:
			return self.sample(u1, u2)

		# sample inside subtended cone
		st_max_sq = self.radius * self.radius / pnt.sq_dist(ctr)
		ct_max = np.sqrt(max(0., 1. - st_max_sq))

		r = Ray(pnt, uniform_sample_cone(u1, u2, ct_max, wc_x, wc_y, wc), EPS)
		hit, thit, _, _ = self.intersect(r)

		if not hit:
			thit = (ctr - pnt).dot(normalize(r.d))

		ps = r(thit)
		ns = Normal.fromVector(normalize(ps - ctr))
		if self.ro:
			ns *= -1.

		return [ps, ns]

	def pdf_p(self, pnt: 'Point', wi: 'Vector') -> FLOAT:
		ctr = self.o2w(Point(0., 0., 0.))
		# return uniform weight if inside
		if pnt.sq_dist(ctr) - self.radius * self.radius < EPS:
			return super().pdf_p(pnt, wi)

		# general weight
		st_max_sq = self.radius * self.radius / pnt.sq_dist(ctr)
		ct_max = np.sqrt(max(0., 1. - st_max_sq))
		return uniform_cone_pdf(ct_max)

	def intersect(self, r: 'Ray') -> [bool, FLOAT, FLOAT, 'DifferentialGeometry']:
		"""
		Returns:
		 - bool: specify whether intersects
		 - tHit: hit point param
		 - rEps: error tolerance
		 - dg: DifferentialGeometry object
		"""
		# transform ray to object space
		ray = self.w2o(r)

		# solve quad eqn
		A = ray.d.sq_length()
		B = 2 * ray.d.dot(ray.o)
		C = ray.o.sq_length() - self.radius * self.radius

		D = B * B - 4. * A * C
		if D <= 0.:
			return (False, None, None, None)

		# validate solutions
		[t0, t1] = np.roots([A,B,C])
		if t0 > t1:
			t0, t1 = t1, t0
		if t0 > ray.maxt or t1 < ray.mint:
			return (False, None, None, None)
		thit = t0
		if t0 < ray.mint:
			thit = t1
			if thit > ray.maxt:
				return (False, None, None, None)

		# sphere hit position
		phit = ray(thit)
		if phit.x == 0. and phit.y == 0.:
			phit.x = EPS * self.radius
		phi = np.arctan2(phit.y, phit.x)
		if phi < 0.:
			phi += 2. * np.pi

		# test intersection against clipping params
		if (self.zmin > -self.radius and phit.z < self.zmin) or \
				(self.zmax < self.radius and phit.z > self.zmax) or \
				phi > self.phiMax:
			if thit == t1 or t1 > ray.maxt:
				return (False, None, None, None)

			# try again with t1
			thit = t1
			phit = ray(thit)
			if phit.x == 0. and phit.y == 0.:
				phit.x = EPS * self.radius
			phi = np.arctan2(phit.y, phit.x)
			if phi < 0.:
				phi += 2. * np.pi
			if (self.zmin > -self.radius and phit.z < self.zmin) or \
					(self.zmax < self.radius and phit.z > self.zmax) or \
					phi > self.phiMax:
				return (False, None, None, None)

		# otherwise ray hits the sphere
		# initialize the differential structure
		u = phi / self.phiMax
		theta = np.arccos(np.clip(phit.z / self.radius, -1., 1.))
		delta_theta = self.thetaMax - self.thetaMin
		v = (theta - self.thetaMin) * delta_theta

		# find derivatives
		dpdu = Vector(-self.phiMax * phit.y, self.phiMax * phit.x, 0.)
		zrad = np.sqrt(phit.x * phit.x + phit.y * phit.y)
		inv_zrad = 1. / zrad
		cphi = phit.x * inv_zrad
		sphi = phit.y * inv_zrad
		dpdv = delta_theta \
					* Vector(phit.z * cphi, phit.z * sphi, -self.radius * np.sin(theta))

		# derivative of Normals
		# given by Weingarten Eqn
		d2pduu = -self.phiMax * self.phiMax * Vector(phit.x, phit.y, 0.)
		d2pduv = delta_theta * phit.z * self.phiMax * Vector(-sphi, cphi, 0.)
		d2pdvv = -delta_theta * delta_theta * Vector(phit.x, phit.y, phit.z)

		# fundamental forms
		E = dpdu.dot(dpdu)
		F = dpdu.dot(dpdv)
		G = dpdv.dot(dpdv)
		N = normalize(dpdu.cross(dpdv))
		e = N.dot(d2pduu)
		f = N.dot(d2pduv)
		g = N.dot(d2pdvv)

		invEGFF = 1. / (E * G - F * F)
		dndu = Normal.fromVector((f * F - e * G) * invEGFF * dpdu +
								 (e * F - f * E) * invEGFF * dpdv)
		dndv = Normal.fromVector((g * F - f * G) * invEGFF * dpdu +
								 (f * F - g * E) * invEGFF * dpdv)

		o2w = self.o2w
		from src.geometry.diffgeom import DifferentialGeometry
		dg = DifferentialGeometry(o2w(phit), o2w(dpdu), o2w(dpdv), o2w(dndu), o2w(dndv), u, v, self)
		return True, thit, EPS * thit, dg


	def intersectP(self, r: 'Ray') -> bool:	
		# transform ray to object space
		ray = self.w2o(r)

		# solve quad eqn
		A = ray.d.sq_length()
		B = 2 * ray.d.dot(ray.o)
		C = ray.o.sq_length() - self.radius * self.radius

		D = B * B - 4. * A * C
		if D <= 0.:
			return False

		# validate solutions
		[t0, t1] = np.roots([A,B,C])
		if t0 > t1:
			t0, t1 = t1, t0
		if t0 > ray.maxt or t1 < ray.mint:
			return False
		thit = t0
		if t0 < ray.mint:
			thit = t1
			if thit > ray.maxt:
				return False

		phit = ray(thit)
		if phit.x == 0. and phit.y == 0.:
			phit.x = EPS * self.radius
		phi = np.arctan2(phit.y, phit.x)
		if phi < 0.:
			phi += 2. * np.pi

		# test intersection against clipping params
		if (self.zmin > -self.radius and phit.z < self.zmin) or \
				(self.zmax < self.radius and phit.z > self.zmax) or \
				phi > self.phiMax:
			if thit == t1 or t1 > ray.maxt:
				return False

			# try again with t1
			thit = t1
			phit = ray(thit)
			if phit.x == 0. and phit.y == 0.:
				phit.x = EPS * self.radius
			phi = np.arctan2(phit.y, phit.x)
			if phi < 0.:
				phi += 2. * np.pi
			if (self.zmin > -self.radius and phit.z < self.zmin) or \
					(self.zmax < self.radius and phit.z > self.zmax) or \
					phi > self.phiMax:
				return False

		return True				
	
	def area(self) -> FLOAT:
		return self.phiMax * self.radius * (self.zmax - self.zmin)

class Cylinder(Shape):
	"""
	Cylinder Class

	Subclasses `Shape` and is used
	to model possibly partial Cylinder.
	"""	
	def __init__(self, o2w: 'Transform', w2o :'Transform',
			ro: bool, rad: FLOAT, z0: FLOAT, z1: FLOAT, pm: FLOAT):
		super().__init__(o2w, w2o, ro)
		self.radius = rad
		self.zmin = min(z0, z1)
		self.zmax = max(z0, z1)
		self.phiMax = np.deg2rad(np.clip(pm, 0., 360.))

	def __repr__(self):
		return "{}\nRadius: {}".format(self.__class__, self.radius)

	def object_bound(self) -> 'BBox':
		return BBox(Point(-self.radius, -self.radius, self.zmin),
					Point(self.radius, self.radius, self.zmax))

	def sample(self, u1: FLOAT, u2: FLOAT) -> ['Point', 'Normal']:
		z = Lerp(u1, self.zmin, self.zmax)
		t = u2 * self.phiMax
		p = Point(self.radius * np.cos(t), self.radius * np.sin(t), z)
		Ns = normalize(self.o2w(Normal(p.x, p.y, 0.)))

		if self.ro:
			Ns *= -1.

		return [self.o2w(p), Ns]


	def intersect(self, r: 'Ray') -> (bool, FLOAT, FLOAT, 'DifferentialGeometry'):
		"""
		Returns:
		 - bool: specify whether intersects
		 - tHit: hit point param
		 - rEps: error tolerance
		 - dg: DifferentialGeometry object
		"""
		# transform ray to object space
		ray = self.w2o(r)

		# solve quad eqn
		A = ray.d.x * ray.d.x + ray.d.y * ray.d.y
		B = 2 * (ray.d.x * ray.o.x + ray.d.y * ray.o.y)
		C = ray.o.x * ray.o.x + ray.o.y * ray.o.y - self.radius * self.radius

		D = B * B - 4. * A * C
		if D <= 0.:
			return (False, None, None, None)

		# validate solutions
		[t0, t1] = np.roots([A,B,C])
		if t0 > t1:
			t0, t1 = t1, t0
		if t0 > ray.maxt or t1 < ray.mint:
			return (False, None, None, None)
		thit = t0
		if t0 < ray.mint:
			thit = t1
			if thit > ray.maxt:
				return (False, None, None, None)

		# cylinder hit position
		phit = ray(thit)
		phi = np.arctan2(phit.y, phit.x)
		if phi < 0.:
			phi += 2. * np.pi

		# test intersection against clipping params
		if phit.z < self.zmin or phit.z > self.zmax or phi > self.phiMax:
			if thit == t1 or t1 > ray.maxt:
				return (False, None, None, None)

			# try again with t1
			thit = t1
			phit = ray(thit)
			phi = np.arctan2(phit.y, phit.x)
			if phi < 0.:
				phi += 2. * np.pi

			if phit.z < self.zmin or phit.z > self.zmax or phi > self.phiMax:
				if thit == t1 or t1 > ray.maxt:
					return (False, None, None, None)

		# otherwise ray hits the cylinder
		# initialize the differential structure
		u = phi / self.phiMax
		v = (phit.z - self.zmin) / (self.thetaMax - self.thetaMin)

		# find derivatives
		dpdu = Vector(-self.phiMax * phit.y, self.phiMax * phit.x, 0.)
		dpdv = Vector(0., 0., self.zmax - self.zmin)
		
		# derivative of Normals
		# given by Weingarten Eqn
		d2pduu = -self.phiMax * self.phiMax * Vector(phit.x, phit.y, 0.)
		d2pduv = Vector(0., 0., 0.)
		d2pdvv = Vector(0., 0., 0.)

		# fundamental forms
		E = dpdu.dot(dpdu)
		F = dpdu.dot(dpdv)
		G = dpdv.dot(dpdv)
		N = normalize(dpdu.cross(dpdv))
		e = N.dot(d2pduu)
		f = N.dot(d2pduv)
		g = N.dot(d2pdvv)

		invEGFF = 1. / (E * G - F * F)
		dndu = Normal.fromVector((f * F - e * G) * invEGFF * dpdu +
								 (e * F - f * E) * invEGFF * dpdv)
		dndv = Normal.fromVector((g * F - f * G) * invEGFF * dpdu +
								 (f * F - g * E) * invEGFF * dpdv)

		o2w = self.o2w
		dg = DifferentialGeometry(o2w(phit), o2w(dpdu), o2w(dpdv),
								  o2w(dndu), o2w(dndv), u, v, self)
		return True, thit, EPS * thit, dg


	def intersectP(self, r: 'Ray') -> bool:	
		# transform ray to object space
		ray = self.w2o(r)

		# solve quad eqn
		A = ray.d.x * ray.d.x + ray.d.y * ray.d.y
		B = 2 * (ray.d.x * ray.o.x + ray.d.y * ray.o.y)
		C = ray.o.x * ray.o.x + ray.o.y * ray.o.y - self.radius * self.radius

		D = B * B - 4. * A * C
		if D <= 0.:
			return False

		# validate solutions
		[t0, t1] = np.roots([A,B,C])
		if t0 > t1:
			t0, t1 = t1, t0
		if t0 > ray.maxt or t1 < ray.mint:
			return False
		thit = t0
		if t0 < ray.mint:
			thit = t1
			if thit > ray.maxt:
				return False

		# cylinder hit position
		phit = ray(thit)
		phi = np.arctan2(phit.y, phit.x)
		if phi < 0.:
			phi += 2. * np.pi

		# test intersection against clipping params
		if phit.z < self.zmin or phit.z > self.zmax or phi > self.phiMax:
			if thit == t1 or t1 > ray.maxt:
				return False

			# try again with t1
			thit = t1
			phit = ray(thit)
			phi = np.arctan2(phit.y, phit.x)
			if phi < 0.:
				phi += 2. * np.pi

			if phit.z < self.zmin or phit.z > self.zmax or phi > self.phiMax:
				if thit == t1 or t1 > ray.maxt:
					return False

		return True

	
	def area(self) -> FLOAT:
		return self.phiMax * self.radius * (self.zmax - self.zmin)

class Disk(Shape):
	"""
	Disk Class

	Subclasses `Shape` and is used
	to model possibly partial Disk.
	"""	
	def __init__(self, o2w: 'Transform', w2o :'Transform',
			ro: bool, ht: FLOAT, r: FLOAT, ri: FLOAT, pm: FLOAT):
		super().__init__(o2w, w2o, ro)
		self.height = ht
		self.radius = r
		self.inner_radius = ri
		self.phiMax = np.deg2rad(np.clip(pm, 0., 360.))

	def __repr__(self):
		return "{}\nRadius: {}\nInner Radius: {}" \
			.format(self.__class__, self.radius, self.inner_radius)

	def object_bound(self) -> 'BBox':
		return BBox(Point(-self.radius, -self.radius, self.height),
					Point(self.radius, self.radius, self.height))

	def sample(self, u1: FLOAT, u2: FLOAT) -> ['Point', 'Normal']:

		# account for partial disk
		x, y = concentric_sample_disk(u1, u2)
		phi = np.arctan2(y, x) * self.phiMax * INV_2PI
		r = self.inner + np.sqrt(x * x + y * y) * (self.radius - self.inner)

		p = Point(r * np.cos(phi), r * np.sin(phi), self.height)

		Ns = normalize(self.o2w(p))
		if self.ro:
			Ns *= -1.

		return [self.o2w(p), Ns]

	def intersect(self, r: 'Ray') -> (bool, FLOAT, FLOAT, 'DifferentialGeometry'):
		"""
		Returns:
		 - bool: specify whether intersects
		 - tHit: hit point param
		 - rEps: error tolerance
		 - dg: DifferentialGeometry object
		"""
		# transform ray to object space
		ray = self.w2o(r)

		# parallel ray has not intersection
		if np.fabs(ray.d.z) < EPS:
			return (False, None, None, None)

		thit = (self.height - ray.o.z) / ray.d.z
		if thit < ray.mint or thit > ray.maxt:
			return (False, None, None, None)

		phit = ray(thit)

		# check radial distance
		dt2 = phit.x * phit.x + phit.y * phit.y
		if dt2 > self.radius * self.radius or \
				dt2 < self.inner_radius * self.inner_radius:
			return (False, None, None, None)

		# check angle
		phi = np.arctan2(phit.y, phit.x)
		if phi < 0:
			phi += 2. * np.pi

		if phi > self.phiMax:
			return (False, None, None, None)

		# otherwise ray hits the disk
		# initialize the differential structure
		u = phi / self.phiMax
		v = 1. - ((np.sqrt(dt2 - self.inner_radius)) /
				  (self.radius - self.inner_radius))

		# find derivatives
		dpdu = Vector(-self.phiMax * phit.y, self.phiMax * phit.x, 0.)
		dpdv = Vector(-phit.x / (1. - v), -phit.y / (1. - v), 0.)
		dpdu *= self.phiMax * INV_2PI
		dpdv *= (self.radius - self.inner_radius) / self.radius
		
		# derivative of Normals
		dndu = Normal(0., 0., 0.,)
		dndv = Normal(0., 0., 0.,)

		o2w = self.o2w
		dg = DifferentialGeometry(o2w(phit), o2w(dpdu), o2w(dpdv),
								  o2w(dndu), o2w(dndv), u, v, self)
		return True, thit, EPS * thit, dg


	def intersectP(self, r: 'Ray') -> bool:	
		# transform ray to object space
		ray = self.w2o(r)

		# parallel ray has not intersection
		if np.fabs(ray.d.z) < EPS:
			return False

		thit = (self.height - ray.o.z) / ray.d.z
		if thit < ray.mint or thit > ray.maxt:
			return False

		phit = ray(thit)

		# check radial distance
		dt2 = phit.x * phit.x + phit.y * phit.y
		if dt2 > self.radius * self.radius or \
				dt2 < self.inner_radius * self.inner_radius:
			return False

		# check angle
		phi = np.arctan2(phit.y, phit.x)
		if phi < 0:
			phi += 2. * np.pi

		if phi > self.phiMax:
			return False

		return True

	
	def area(self) -> FLOAT:
		return self.phiMax * .5 * \
			(self.radius * self.radius - self.inner_radius * self.inner_radius)




