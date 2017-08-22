"""
loopsubdiv.py

Loop subdividing surface implementation.

Created by Jiayao on July 27, 2017
Modified on Aug 13, 2017
"""
from __future__ import absolute_import
from pytracer import *
import pytracer.geometry as geo
import pytracer.transform as trans
from pytracer.shape import Shape, NEXT, PREV


__all__ = ['create_loop_subdiv', 'LoopSubdiv']


def create_loop_subdiv(o2w: 'trans.Transform', w2o: 'trans.Transform',
                       ro: bool, params: {str: object}):
	nlevels = 3 if 'nlevels' not in params else params['nlevels']
	vi = None if 'indices' not in params else params['indices']
	P = None if 'P' not in params else params['P']

	if vi is None or P is None:
		return None

	if not len(P) % 3 == 0:
		util.logging('Error', 'LoopSubDiv: data error')
		return None

	npnt = len(P) // 3

	p = [None] * npnt
	cnt = 0
	for i in range(0, len(P), 3):
		p[cnt] = geo.Point(P[i], P[i+1], P[i+2])
		cnt += 1

	return LoopSubdiv(o2w, w2o, ro, len(vi) // 3, npnt, vi, p, nlevels)


class LoopSubdiv(Shape):
	"""
	LoopSubdiv Class

	Implement Loop's method for subdivision surfaces.
	Assumes meshes are manifolds and are consistently ordered.
	"""
	class SDVertex(object):
		"""
		SDVertex Class

		Inner class for `LoopSubdiv`
		"""

		def __init__(self, pt: 'geo.Point' = geo.Point(0., 0., 0., )):
			self.P = pt.copy()
			self.startFace = None  # ref to SDFace
			self.child = None  # ref to SDVertex
			self.regular = self.boundary = False

		def __repr__(self):
			return "SDVertex: {}\nPoint: {}".format(id(self), self.P)

		def __lt__(self, other):
			return id(self) < id(other)

		def __eq__(self, other):
			return id(self) == id(other)

		def __hash__(self):
			return id(self)

		def valence(self):
			f = self.startFace
			if not self.boundary:
				# interior
				nf = 1
				f = f.next_face(self)
				while f != self.startFace:
					nf += 1
					f = f.next_face(self)
				return nf
			else:
				# boundary
				nf = 1
				f = f.next_face(self)
				while f is not None:
					nf += 1
					f = f.next_face(self)
				f = self.startFace
				f = f.prev_face(self)
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

	class SDFace(object):
		"""
		SDFace Class

		Inner class for `LoopSubdiv`
		"""

		def __init__(self):
			self.v = [None, None, None]  # ref to SDVertex (3)
			self.f = [None, None, None]  # ref to SDFace (3)
			self.children = [None, None, None, None]  # ref to SDFace (4)

		def __repr__(self):
			return "SDFace: {}\n".format(id(self))

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
			return self.v[NEXT(self.vnum(vert))]

		def prev_vert(self, vert):
			return self.v[PREV(self.vnum(vert))]

		def other_vert(self, v0, v1):
			for i in range(3):
				if self.v[i] != v0 and self.v[i] != v1:
					return self.v[i]
			raise RuntimeError('Logic error in Shape.LoopSubdiv.SDVertex.other_vert()')

	class SDEdge(object):
		"""
		SDEdge Class

		Inner class for `LoopSubdiv`
		"""

		def __init__(self, v0: 'SDVertex' = None, v1: 'SDVertex' = None):
			self.v = [v0, v1] if id(v0) < id(v1) else [v1, v0]
			self.f = [None, None]
			self.f0edge_num = -1

		def __repr__(self):
			return "SDEdge: {}\n".format(id(self))

		def __lt__(self, other):
			if id(self.v[0]) == id(other.v[0]):
				return id(self.v[1]) < id(other.v[1])
			return id(self.v[0]) < id(other.v[0])

		def __eq__(self, other):
			return id(self.v[0]) == id(other.v[0]) and id(self.v[1]) == id(other.v[1])

		def __hash__(self):
			return hash( (id(self.v[0]), id(self.v[1])) )

	def __init__(self, o2w: 'trans.Transform', w2o: 'trans.Transform',
	             ro: bool, nf: INT, nv: INT, vi: [INT],
	             P: ['geo.Point'], nl: INT):
		super().__init__(o2w, w2o, ro)
		self.nLevels = nl
		self.vertices = np.array([LoopSubdiv.SDVertex(P[i]) for i in range(nv)])
		self.faces = np.array([self.SDFace() for _ in range(nf)])

		# set face to vertex refs
		k = 0
		for i in range(nf):
			f = self.faces[i]
			for j in range(3):
				f.v[j] = self.vertices[vi[k + j]]
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
					del edges[e]  # by assumption, each edge appears at most twice
				else:
					# new edge
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
			v.boundary = (f is None)

			val = v.valence()

			if not v.boundary and val == 6:
				v.regular = True
			elif v.boundary and val == 4:
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

	def weight_one_ring(self, vert: 'SDVertex', beta: 'FLOAT') -> 'geo.Point':
		valence = vert.valence()
		p_ring = vert.one_ring()
		P = (1. - valence * beta) * vert.P
		for p in p_ring:
			P += beta * p
		return P

	def weight_boundary(self, vert: 'SDVertex', beta: 'FLOAT') -> 'geo.Point':
		valence = vert.valence()
		p_ring = vert.one_ring()
		P = (1. - 2. * beta) * vert.P
		P += beta * (p_ring[0] + p_ring[-1])
		return P

	def object_bound(self) -> 'geo.BBox':
		ret = geo.BBox()
		for i in range(len(self.vertices)):
			ret.union(self.vertices[i].P)
		return ret

	def world_bound(self) -> 'geo.BBox':
		ret = geo.BBox()
		for i in range(len(self.vertices)):
			ret.union(self.o2w(self.vertices[i].P))
		return ret

	def can_intersect(self) -> bool:
		return False

	def refine(self) -> ['Shape']:
		f = self.faces
		v = self.vertices
		for i in range(self.nLevels):
			new_faces = []
			new_vertices = []
			# update next level
			for vv in v:
				vv.child = LoopSubdiv.SDVertex()
				vv.child.regular = vv.regular
				vv.child.boundary = vv.boundary
				new_vertices.append(vv.child)

			for ff in f:
				for k in range(4):
					ff.children[k] = LoopSubdiv.SDFace()
					new_faces.append(ff.children[k])

			# update vertex positions and create new edge vertices
			# update for even vertices (already presented)
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

			# update for odd vertices (along the split edges)
			edge_verts = {}
			for ff in f:
				for k in range(3):
					# k-th edge
					edge = LoopSubdiv.SDEdge(ff.v[k], ff.v[NEXT(k)])

					if edge not in edge_verts:
						# create and init
						vert = LoopSubdiv.SDVertex()
						new_vertices.append(vert)
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
						edge_verts[edge] = vert

			# update new mesh topology
			# startFace ref of odd vertices (done)
			# startFace ref of even vertices
			for vv in v:
				v_num = vv.startFace.vnum(v)
				vv.child.startFace = vv.startFace.child[v_num]

			# new faces' neighbor ref
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
					vert = edge_verts[LoopSubdiv.SDEdge(ff.v[k], ff.v[NEXT(k)])]
					ff.children[k].v[NEXT(k)] = vert
					ff.children[NEXT(k)].v[k] = vert
					ff.children[3].v[k] = vert

			# prepare for next level
			f = np.array(new_faces)
			v = np.array(new_vertices)

		# add vertices to the limiting surface
		Plimit = []  # [geo.Point]
		for vv in v:
			if vv.boundary:
				Plimit.append(self.weight_boundary(vv, .2))
			else:
				Plimit.append(self.weight_one_ring(vv, self.gamma(vv.valence())))

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
				S = Pring[valence - 1] - Pring[0]
				if valence == 2:
					T = geo.Vector.fromPoint(Pring[0] + Pring[1] - 2 * vert.P)
				elif valence == 3:
					T = Pring[1] - v.P
				elif valence == 4:
					T = geo.Vector(
						-Pring[0] + 2 * Pring[1] + 2 * Pring[2] + -Pring[3] + -2 * vv.P)  # avoid geo.Point substraction
				else:
					theta = np.pi / (valence - 1)
					T = geo.Vector(np.sin(theta) * (Pring[0] + Pring[-1]))
					T += np.sum([((2 * np.cos(theta) - 2) * np.sin(k * theta)) * \
					             Pring[k] for k in range(valence - 1)], axis=0)
					T = -T

			Ns.append(geo.Normal.fromVector(S.cross(T)))

		# create `TriangleMesh`
		verts = []
		usedVerts = {}
		for i, vv in enumerate(v):
			usedVerts[vv] = i
		for ff in f:
			for j in range(3):
				verts.append(usedVerts[ff.v[j]])

		params = {'indices': verts,
		          'P': Plimit,
		          'N': Ns}

		from .triangle import create_triangle_mesh
		return [create_triangle_mesh(self.o2w, self.w2o, self.ro, params)]

	def intersect(self, r: 'geo.Ray') -> (bool, FLOAT, FLOAT, 'geo.DifferentialGeometry'):
		raise NotImplementedError('{} cannot intersect before refinement'.format(self.__class__))

	def intersect_p(self, r: 'geo.Ray') -> bool:
		raise NotImplementedError('{} cannot intersect before refinement'.format(self.__class__))

	def area(self) -> FLOAT:
		raise NotImplementedError('unimplemented Shape.area() method called')


