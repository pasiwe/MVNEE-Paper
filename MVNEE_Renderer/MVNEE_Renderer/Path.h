#pragma once

#include "Settings.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using glm::vec3;
using namespace std;

enum VertexType {
	TYPE_ORIGIN,
	TYPE_MEDIUM_SCATTERING,
	TYPE_SURFACE,
	TYPE_MVNEE
};

enum VertexMediumType {
	VERT_NO_MEDIUM,
	VERT_HOM_MEDIUM,
	VERT_HET_MEDIUM
};

struct PathVertex {
	vec3 vertex;
	vec3 surfaceNormal;
	VertexType vertexType;
	VertexMediumType vertexMediumType;
	int geometryID;
	double density;

	PathVertex() : vertex(0.0f), vertexType(TYPE_MEDIUM_SCATTERING), geometryID(-1), surfaceNormal(vec3(0.0f)), density(0.0), vertexMediumType(VERT_HOM_MEDIUM) {

	}

	PathVertex(const vec3& vertex, const VertexType& vertexType, int geometryID, vec3 surfaceNormal, VertexMediumType vertexMediumType) : vertex(vertex),
		vertexType(vertexType),
		geometryID(geometryID),
		surfaceNormal(surfaceNormal),
		density(0.0f),
		vertexMediumType(vertexMediumType)
	{

	}

	PathVertex(const vec3& vertex, const VertexType& vertexType, int geometryID, vec3 surfaceNormal, double density, VertexMediumType vertexMediumType) : vertex(vertex),
		vertexType(vertexType),
		geometryID(geometryID),
		surfaceNormal(surfaceNormal),
		density(density),
		vertexMediumType(vertexMediumType)
	{

	}
};

/**
*	Path that stores the vertices of the path.
*/
class Path
{

private:

	const int MAX_SEGMENT_COUNT;

	/** Number of segments (vertex-to-vertex-conenction) of the path */
	int segmentLength;

	/** Number of actual vertices in the path*/
	int vertexCount;

	int pathTracingVertexCount;

	/** Maximum number of vertices for this path */
	int maxPathVertices;

	/** Vertices of the path. Starting at the image plane vertex, ending on a light vertex. */
	//PathVertex pathVertices[RenderingSettings::MAX_SEGMENT_COUNT + 1];
	PathVertex* pathVertices;

public:
	Path(const int MAX_SEGMENT_COUNT) : MAX_SEGMENT_COUNT(MAX_SEGMENT_COUNT)
	{
		segmentLength = -1;
		maxPathVertices = MAX_SEGMENT_COUNT + 1;
		pathVertices = new PathVertex[maxPathVertices];

		vertexCount = 0;
		pathTracingVertexCount = 0;
	}

	Path(const PathVertex& startVertex, const int MAX_SEGMENT_COUNT) : MAX_SEGMENT_COUNT(MAX_SEGMENT_COUNT)
	{
		segmentLength = 0;
		maxPathVertices = MAX_SEGMENT_COUNT + 1;

		pathVertices = new PathVertex[maxPathVertices];

		vertexCount = 0;
		pathVertices[vertexCount] = startVertex;
		vertexCount++;
		pathTracingVertexCount = 1;
	}

	~Path()
	{
		delete[] pathVertices;
	}

	/* Resets all attributes, so the Path can be reused without having to create a new instance */
	void reset() {
		segmentLength = -1;
		vertexCount = 0;
		pathTracingVertexCount = 0;
	}

	inline void addVertex(const PathVertex& nextPathVertex) {
		if (vertexCount < maxPathVertices) {
			pathVertices[vertexCount] = nextPathVertex;
			vertexCount++;
			segmentLength++;
			if (nextPathVertex.vertexType != TYPE_MVNEE) {
				pathTracingVertexCount++;
				//make sure the last path vertex was NOT an MVNEE type vertex, that would be illegal!!!
				if (segmentLength > 1) {
					if (pathVertices[segmentLength - 1].vertexType == TYPE_MVNEE) {
						cout << "[Path] standard vertex added after MVNEE vert!!!!" << endl;
						assert(pathVertices[segmentLength - 1].vertexType != TYPE_MVNEE);
					}
				}
			} 
		}
		else {
			cout << "Path Array is full!" << endl;
		}
	}

	inline void addHomVolVertex(const vec3& vertex, const VertexType& type) {
		assert(type != TYPE_SURFACE);
		PathVertex newPathVertex = PathVertex(vertex, type, -1, vec3(0.0f), VERT_HOM_MEDIUM);
		addVertex(newPathVertex);		
	}

	inline void addHetVolVertex(const vec3& vertex, const VertexType& type, double density) {
		assert(type != TYPE_SURFACE);
		PathVertex newPathVertex = PathVertex(vertex, type, -1, vec3(0.0f), density, VERT_HET_MEDIUM);
		addVertex(newPathVertex);
	}

	inline void addSurfaceVertex(const vec3& vertex, const int geometryID, const vec3& surfaceNormal, const VertexMediumType vertexMediumType) {
		if (vertexCount < maxPathVertices) {
			pathVertices[vertexCount] = PathVertex(vertex, TYPE_SURFACE, geometryID, surfaceNormal, vertexMediumType);
			vertexCount++;
			segmentLength++;
			pathTracingVertexCount++;
		}
		else {
			cout << "Path Array is full!" << endl;
		}
	}

	inline PathVertex* getPathVertices() {
		return pathVertices;
	}

	inline int getSegmentLength()
	{
		return segmentLength;
	}

	inline int getVertexCount()
	{
		return vertexCount;
	}

	inline VertexType getTypeAt(int index) {
		assert(index < vertexCount && index >= 0);
		return pathVertices[index].vertexType;
	}

	inline void getVertex(int index, PathVertex& output) {
		assert(index < vertexCount);
		if (index < vertexCount) {
			output = pathVertices[index];
		}
		else {
			cout << "Path: index out of bounds!" << endl;
		}
	}

	inline vec3 getVertexPosition(int index) {
		assert(index < vertexCount);
		if (index < vertexCount) {
			return pathVertices[index].vertex;
		}
		else {
			cout << "Path: index out of bounds!" << endl;
			return vec3(0.0f);
		}
	}

	/**
	* Cut the end MVNEE vertices by reducing the segment length and vertex count to the path tracing vertex count.
	*/
	inline void cutMVNEEVertices() {
		vertexCount = pathTracingVertexCount;
		segmentLength = pathTracingVertexCount - 1;

	}

	/**
	* Cut the end vertices by reducing the segment length and vertex count.
	*/
	void reduceSegmentLengthTo(int newSegmentLength) {
		segmentLength = newSegmentLength;
		vertexCount = newSegmentLength + 1;
	}
};