#pragma once
#define _USE_MATH_DEFINES

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include <iostream>

#include <math.h>

#include "StringParser.h"

#include <glm/glm.hpp>
#include <vector>

#include <embree2/rtcore.h>
#include <embree2/rtcore_ray.h>
#include <embree2/rtcore_geometry.h>
#include <embree2/rtcore_scene.h>

#include "Camera.h"
#include "LightSource.h"
#include "Settings.h"
#include "BSDF.h"

#include <rapidXML/rapidxml.hpp>
#include <rapidXML/rapidxml_utils.hpp>

using namespace std;
using namespace rapidxml;
using glm::vec3;

struct vector3{ float x, y, z; };
struct EmbreeVertex   { float x, y, z, r; };
struct Triangle { int v0, v1, v2; };
struct Quad     { int v0, v1, v2, v3; };

struct ObjectData{ string name; BSDF* bsdf; };

class Scene
{

private:
	RTCScene scene; //Scene object for embree	
	//ObjectData objectData[MAX_OBJECT_COUNT];
	vector<ObjectData> objectData;

	double* lightFluxes;
	double lightFluxSum;

	LightChoiceStrategy lightChoiceStrategy;

	/*
	* Light Sampling function pointer: the Light Sampling function is chosen at run time.
	* All Light Sampling functions have to return a vertex on the chosen light source, as well as the index of the chosen light source.
	* The parameters are the distances to all lightSources, as well as 3 uniform random variables xi in [0,1)
	*/
	vec3(Scene::*lightSampler)(float* lightDistancesArray, const double& xi1, const double& xi2, const double& xi3, int* lightIndex) = NULL;

	/*
	* Light Vertex PDF function pointer: the Light Sampling function is chosen at run time.
	* All Light Vertex PDF functions have to return a vertex on the chosen light source.
	* The parameters are the distances to all lightSources, as well as the light index used for sampling.
	*/
	double(Scene::*lightVertexPDF)(float* lightDistancesArray, int lightIndex) = NULL;


public:
	Camera camera;

	//Multiple Light Information
	int lightSourceCount;
	LightSource** lightSources;	
	
public:

	Scene(RTCScene scene) : scene(scene) {
		lightFluxSum = 0.0;

		//default light sampling strategy and PDF calculation:
		lightChoiceStrategy = UNIFORM;
		lightSampler = &Scene::sampleLightPositionUniformly;
		lightVertexPDF = &Scene::getLightPositionSamplingPDFUniform;

		objectData = vector<ObjectData>();
	}

	~Scene() {
		for (int i = 0; i < lightSourceCount; i++) {
			delete lightSources[i];
		}
		delete[] lightSources;
		delete[] lightFluxes;

		for (int i = 0; i < (int)objectData.size(); i++) {
			objectData[i].bsdf->deleteBSDF();
		}
		objectData.clear();
	}

	void initializeLights(const vector<LightSource*>& lightSourceContainer)
	{
		lightSourceCount = (int)lightSourceContainer.size();
		lightSources = new LightSource*[lightSourceCount];
		lightFluxes = new double[lightSourceCount];
		for (int i = 0; i < lightSourceCount; i++) {
			LightSource* currLight = lightSourceContainer[i];
			lightSources[i] = currLight;
			double lightFlux = (double)currLight->lightBrightness;
			lightFluxes[i] = lightFlux;
			lightFluxSum += lightFlux;
		}
	}


	/**
	* First samples a light source, then a vertex on that light source. The sampled vertex, as well as the index of the chosen light source is returned.
	* An index of -1 indicates an error and should be handled appropriately.
	*/
	inline vec3 sampleLightPosition(float* lightDistancesArray, const double& xi1, const double& xi2, const double& xi3, int* lightIndex)
	{
		if (lightSourceCount == 1) {
			*lightIndex = 0;
			return lightSources[0]->sampleLightPosition(xi1, xi2);
		}
		else {
			return (*this.*lightSampler)(lightDistancesArray, xi1, xi2, xi3, lightIndex);
		}
	}

	inline double getLightVertexSamplingPDF(float* lightDistancesArray, int lightIndex)
	{
		if (lightSourceCount == 1) {
			assert(lightIndex == 0);
			return lightSources[0]->getPositionSamplingPDF();
		}
		else {
			return (*this.*lightVertexPDF)(lightDistancesArray, lightIndex);
		}
	}

	
	inline bool lightIntersected(int geomID, int* intersectedLightSourceIndex)
	{
		for (int i = 0; i < lightSourceCount; i++)
		{
			if (lightSources[i]->lightIntersected(geomID)) {
				*intersectedLightSourceIndex = i;
				return true;
			}
		}

		return false;
	}


	/**
	* Adds a ground plane to the scene .
	* Returns the Embree geometry ID;
	*/
	unsigned int addGroundPlane(const string& name, BSDF* bsdf, const float& sideLength, const float& yPosition)
	{
		/* create a triangulated plane with 2 triangles and 4 vertices */
		unsigned int mesh = rtcNewTriangleMesh(scene, RTC_GEOMETRY_STATIC, 2, 4);

		/* set vertices */
		EmbreeVertex* vertices = (EmbreeVertex*)rtcMapBuffer(scene, mesh, RTC_VERTEX_BUFFER);
		vertices[0].x = -sideLength; vertices[0].y = yPosition; vertices[0].z = -sideLength;
		vertices[1].x = -sideLength; vertices[1].y = yPosition; vertices[1].z = +sideLength;
		vertices[2].x = +sideLength; vertices[2].y = yPosition; vertices[2].z = -sideLength;
		vertices[3].x = +sideLength; vertices[3].y = yPosition; vertices[3].z = +sideLength;
		rtcUnmapBuffer(scene, mesh, RTC_VERTEX_BUFFER);

		/* set triangles */
		Triangle* triangles = (Triangle*)rtcMapBuffer(scene, mesh, RTC_INDEX_BUFFER);
		triangles[0].v0 = 0; triangles[0].v1 = 2; triangles[0].v2 = 1;
		triangles[1].v0 = 1; triangles[1].v1 = 2; triangles[1].v2 = 3;
		rtcUnmapBuffer(scene, mesh, RTC_INDEX_BUFFER);

		ObjectData objectData;
		objectData.bsdf = bsdf;
		objectData.name = name;
		addObjectData(mesh, objectData);

		return mesh;
	}

	unsigned int addCube(const vec3& min, const vec3& max, const string& name, BSDF* bsdf, bool invertNormal)
	{
		/* create a triangulated cube with 12 triangles and 8 vertices */
		unsigned int mesh = rtcNewTriangleMesh(scene, RTC_GEOMETRY_STATIC, 12, 8);

		/* set vertices and vertex colors */
		EmbreeVertex* vertices = (EmbreeVertex*)rtcMapBuffer(scene, mesh, RTC_VERTEX_BUFFER);
		vertices[0].x = min.x; vertices[0].y = min.y; vertices[0].z = min.z;
		vertices[1].x = min.x; vertices[1].y = min.y; vertices[1].z = max.z;
		vertices[2].x = min.x; vertices[2].y = max.y; vertices[2].z = min.z;
		vertices[3].x = min.x; vertices[3].y = max.y; vertices[3].z = max.z;
		vertices[4].x = max.x; vertices[4].y = min.y; vertices[4].z = min.z;
		vertices[5].x = max.x; vertices[5].y = min.y; vertices[5].z = max.z;
		vertices[6].x = max.x; vertices[6].y = max.y; vertices[6].z = min.z;
		vertices[7].x = max.x; vertices[7].y = max.y; vertices[7].z = max.z;
		rtcUnmapBuffer(scene, mesh, RTC_VERTEX_BUFFER);

		/* set triangles and face colors */
		int tri = 0;
		Triangle* triangles = (Triangle*)rtcMapBuffer(scene, mesh, RTC_INDEX_BUFFER);

		if (invertNormal) {
			// left side
			triangles[tri].v0 = 0; triangles[tri].v1 = 1; triangles[tri].v2 = 2; tri++;
			triangles[tri].v0 = 1; triangles[tri].v1 = 3; triangles[tri].v2 = 2; tri++;

			// right side
			triangles[tri].v0 = 4; triangles[tri].v1 = 6; triangles[tri].v2 = 4; tri++;
			triangles[tri].v0 = 5; triangles[tri].v1 = 6; triangles[tri].v2 = 7; tri++;

			// bottom side
			triangles[tri].v0 = 0; triangles[tri].v1 = 4; triangles[tri].v2 = 1; tri++;
			triangles[tri].v0 = 1; triangles[tri].v1 = 4; triangles[tri].v2 = 5; tri++;

			// top side
			triangles[tri].v0 = 2; triangles[tri].v1 = 3; triangles[tri].v2 = 6; tri++;
			triangles[tri].v0 = 3; triangles[tri].v1 = 7; triangles[tri].v2 = 6; tri++;

			// front side
			triangles[tri].v0 = 0; triangles[tri].v1 = 2; triangles[tri].v2 = 4; tri++;
			triangles[tri].v0 = 2; triangles[tri].v1 = 6; triangles[tri].v2 = 4; tri++;

			// back side
			triangles[tri].v0 = 1; triangles[tri].v1 = 5; triangles[tri].v2 = 3; tri++;
			triangles[tri].v0 = 3; triangles[tri].v1 = 5; triangles[tri].v2 = 7; tri++;
		}
		else {
			// left side
			triangles[tri].v0 = 0; triangles[tri].v1 = 2; triangles[tri].v2 = 1; tri++;
			triangles[tri].v0 = 1; triangles[tri].v1 = 2; triangles[tri].v2 = 3; tri++;

			// right side
			triangles[tri].v0 = 4; triangles[tri].v1 = 5; triangles[tri].v2 = 6; tri++;
			triangles[tri].v0 = 5; triangles[tri].v1 = 7; triangles[tri].v2 = 6; tri++;

			// bottom side
			triangles[tri].v0 = 0; triangles[tri].v1 = 1; triangles[tri].v2 = 4; tri++;
			triangles[tri].v0 = 1; triangles[tri].v1 = 5; triangles[tri].v2 = 4; tri++;

			// top side
			triangles[tri].v0 = 2; triangles[tri].v1 = 6; triangles[tri].v2 = 3; tri++;
			triangles[tri].v0 = 3; triangles[tri].v1 = 6; triangles[tri].v2 = 7; tri++;

			// front side
			triangles[tri].v0 = 0; triangles[tri].v1 = 4; triangles[tri].v2 = 2; tri++;
			triangles[tri].v0 = 2; triangles[tri].v1 = 4; triangles[tri].v2 = 6; tri++;

			// back side
			triangles[tri].v0 = 1; triangles[tri].v1 = 3; triangles[tri].v2 = 5; tri++;
			triangles[tri].v0 = 3; triangles[tri].v1 = 7; triangles[tri].v2 = 5; tri++;
		}

		rtcUnmapBuffer(scene, mesh, RTC_INDEX_BUFFER);

		ObjectData objectData;
		objectData.bsdf = bsdf;
		objectData.name = name;
		addObjectData(mesh, objectData);

		return mesh;
	}

	/* adds a cube to the scene */
	unsigned int addCube(const vec3& center, float sideLengthHalf, const string& name, BSDF* bsdf)
	{
		/* create a triangulated cube with 12 triangles and 8 vertices */
		unsigned int mesh = rtcNewTriangleMesh(scene, RTC_GEOMETRY_STATIC, 12, 8);

		/* set vertices and vertex colors */
		EmbreeVertex* vertices = (EmbreeVertex*)rtcMapBuffer(scene, mesh, RTC_VERTEX_BUFFER);
		vertices[0].x = center.x - sideLengthHalf; vertices[0].y = center.y - sideLengthHalf; vertices[0].z = center.z - sideLengthHalf;
		vertices[1].x = center.x - sideLengthHalf; vertices[1].y = center.y - sideLengthHalf; vertices[1].z = center.z + sideLengthHalf;
		vertices[2].x = center.x - sideLengthHalf; vertices[2].y = center.y + sideLengthHalf; vertices[2].z = center.z - sideLengthHalf;
		vertices[3].x = center.x - sideLengthHalf; vertices[3].y = center.y + sideLengthHalf; vertices[3].z = center.z + sideLengthHalf;
		vertices[4].x = center.x + sideLengthHalf; vertices[4].y = center.y - sideLengthHalf; vertices[4].z = center.z - sideLengthHalf;
		vertices[5].x = center.x + sideLengthHalf; vertices[5].y = center.y - sideLengthHalf; vertices[5].z = center.z + sideLengthHalf;
		vertices[6].x = center.x + sideLengthHalf; vertices[6].y = center.y + sideLengthHalf; vertices[6].z = center.z - sideLengthHalf;
		vertices[7].x = center.x + sideLengthHalf; vertices[7].y = center.y + sideLengthHalf; vertices[7].z = center.z + sideLengthHalf;
		rtcUnmapBuffer(scene, mesh, RTC_VERTEX_BUFFER);

		/* set triangles and face colors */
		int tri = 0;
		Triangle* triangles = (Triangle*)rtcMapBuffer(scene, mesh, RTC_INDEX_BUFFER);

		// left side
		triangles[tri].v0 = 0; triangles[tri].v1 = 2; triangles[tri].v2 = 1; tri++;
		triangles[tri].v0 = 1; triangles[tri].v1 = 2; triangles[tri].v2 = 3; tri++;

		// right side
		triangles[tri].v0 = 4; triangles[tri].v1 = 5; triangles[tri].v2 = 6; tri++;
		triangles[tri].v0 = 5; triangles[tri].v1 = 7; triangles[tri].v2 = 6; tri++;

		// bottom side
		triangles[tri].v0 = 0; triangles[tri].v1 = 1; triangles[tri].v2 = 4; tri++;
		triangles[tri].v0 = 1; triangles[tri].v1 = 5; triangles[tri].v2 = 4; tri++;

		// top side
		triangles[tri].v0 = 2; triangles[tri].v1 = 6; triangles[tri].v2 = 3; tri++;
		triangles[tri].v0 = 3; triangles[tri].v1 = 6; triangles[tri].v2 = 7; tri++;

		// front side
		triangles[tri].v0 = 0; triangles[tri].v1 = 4; triangles[tri].v2 = 2; tri++;
		triangles[tri].v0 = 2; triangles[tri].v1 = 4; triangles[tri].v2 = 6; tri++;

		// back side
		triangles[tri].v0 = 1; triangles[tri].v1 = 3; triangles[tri].v2 = 5; tri++;
		triangles[tri].v0 = 3; triangles[tri].v1 = 7; triangles[tri].v2 = 5; tri++;

		rtcUnmapBuffer(scene, mesh, RTC_INDEX_BUFFER);

		ObjectData objectData;
		objectData.bsdf = bsdf;
		objectData.name = name;
		addObjectData(mesh, objectData);

		return mesh;
	}

	/**
	* Adds a circular plane specified by its center and radius, as well as the u and v vectors which together with the normal form a left handed coordinate system.
	* The resulting circualr plane lies in the u-v-plane.
	*/
	unsigned int addCircularPlane(const vec3& center, const vec3& normal, const vec3& u, const vec3& v, const float& radius, const int triangleCount, const string& name, BSDF* bsdf)
	{
		/* create a triangulated circular plane with triangleCount triangles*/
		unsigned int mesh = rtcNewTriangleMesh(scene, RTC_GEOMETRY_STATIC, triangleCount, triangleCount + 1);

		/* set vertices */
		EmbreeVertex* vertices = (EmbreeVertex*)rtcMapBuffer(scene, mesh, RTC_VERTEX_BUFFER);
		//first set center vertex:
		vertices[0].x = center.x; vertices[0].y = center.y; vertices[0].z = center.z;

		float theta;
		vec3 currentVertex;
		for (int i = 0; i < triangleCount; i++) {
			theta = ((float)i / (float)triangleCount) * (float)(2.0 * M_PI);
			currentVertex = center + radius * sinf(theta) * u + radius * cosf(theta) * v;
			vertices[i + 1].x = currentVertex.x; vertices[i + 1].y = currentVertex.y; vertices[i + 1].z = currentVertex.z;
		}
		rtcUnmapBuffer(scene, mesh, RTC_VERTEX_BUFFER);

		/* set triangle faces */
		Triangle* triangles = (Triangle*)rtcMapBuffer(scene, mesh, RTC_INDEX_BUFFER);
		for (int i = 0; i < triangleCount - 1; i++) {
			triangles[i].v0 = 0; triangles[i].v1 = i + 2; triangles[i].v2 = i + 1;
		}
		//last triangle to close to circle:
		triangles[triangleCount - 1].v0 = 0; triangles[triangleCount - 1].v1 = 1; triangles[triangleCount - 1].v2 = triangleCount;
		rtcUnmapBuffer(scene, mesh, RTC_INDEX_BUFFER);

		ObjectData objectData;
		objectData.bsdf = bsdf;
		objectData.name = name;
		addObjectData(mesh, objectData);

		return mesh;
	}


	/**
	* Adds the geometry of the specified obj. file to the scene.
	* Returns the Embree geometry ID;
	*/
	unsigned int addObject(const string& objFilePath, const string& name, BSDF* bsdf, const vec3& translationVector, const float& scaling, const bool& flipZAxis, const bool& flipVertexOrder) {

		cout << "Object: " << name;

		ifstream file;
		file.open(objFilePath, ios::in); // opens as ASCII!
		if (!file) {
			cout << "Object file could not be opened!" << endl;
			cout << "Please check the file path: " << objFilePath << endl;
			return RTC_INVALID_GEOMETRY_ID;
		}

		string currentLine;
		vector<vec3> vertices = vector<vec3>();
		vector<Triangle> faces = vector<Triangle>();

		while (!file.eof()) {
			getline(file, currentLine);
			StringParser line = StringParser(currentLine);
			if (!line.startsWith("#")) {
				//handle all different specifiers
				if (line.startsWith("vn")) {
				}
				else if (line.startsWith("vt")) {
				}
				else if (line.startsWith("v ")) {
					vec3 v = line.getVec3Param("v");
					if (flipZAxis) {
						vec3 vAdjusted = vec3(v.x, v.y, -v.z);
						vec3 transformedVec = (vAdjusted * scaling) + translationVector;
						vertices.push_back(transformedVec);
					}
					else {
						vec3 transformedVec = (v * scaling) + translationVector;
						vertices.push_back(transformedVec);
					}
				}
				else if (line.startsWith("f")) {

					StringParser faceStrings = StringParser(line.retrieveArgumentForParam("f"));
					//first split at ' '
					vector<StringParser> faceList = vector<StringParser>();
					faceStrings.split(' ', &faceList);

					int faceVertexCount = (int)faceList.size();
					//only triangels allowed for now!;
					//assert(faceVertexCount == 3);

					if (faceVertexCount == 3) {

						Triangle currentFace;

						vector<StringParser> perVertexIndices;
						faceList.at(0).split('/', &perVertexIndices);
						int v0 = atoi(perVertexIndices[0].getCharacterData());						

						vector<StringParser> perVertexIndices2;
						faceList.at(1).split('/', &perVertexIndices2);
						int v1 = atoi(perVertexIndices2[0].getCharacterData());						

						vector<StringParser> perVertexIndices3;
						faceList.at(2).split('/', &perVertexIndices3);
						int v2 = atoi(perVertexIndices3[0].getCharacterData());
						

						if (flipVertexOrder) {
							currentFace.v0 = v0 - 1;
							currentFace.v1 = v2 - 1;
							currentFace.v2 = v1 - 1;
						}
						else {
							currentFace.v0 = v0 - 1;
							currentFace.v1 = v1 - 1;
							currentFace.v2 = v2 - 1;
						}

						faces.push_back(currentFace);
					}
					else if (faceVertexCount > 3) {
						vector<int> faceIndices(faceVertexCount);
						for (int i = 0; i < faceVertexCount; i++) {
							vector<StringParser> perVertexIndices;
							faceList.at(i).split('/', &perVertexIndices);
							int v = atoi(perVertexIndices[0].getCharacterData()) - 1;

							faceIndices[i] = v;
						}

						//create triangles
						for (int i = 1; i < (faceVertexCount - 1); i++) {
							Triangle currentFace;
							if (flipVertexOrder) {
								currentFace.v0 = faceIndices[0];
								currentFace.v1 = faceIndices[i + 1];
								currentFace.v2 = faceIndices[i];
							}
							else {
								currentFace.v0 = faceIndices[0];
								currentFace.v1 = faceIndices[i];
								currentFace.v2 = faceIndices[i + 1];
							}
							faces.push_back(currentFace);
						}
					}
				}
			}
		}

		int vertexCount = (int)vertices.size();
		int faceCount = (int)faces.size();

		cout << " has " << faceCount << " triangles." << endl;

		/* create a mesh with faceCount triangles and vertexCount vertices */
		unsigned int mesh = rtcNewTriangleMesh(scene, RTC_GEOMETRY_STATIC, faceCount, vertexCount);

		///////////////
		//read vertices
		///////////////

		/* set vertices */
		EmbreeVertex* verticesBuffer = (EmbreeVertex*)rtcMapBuffer(scene, mesh, RTC_VERTEX_BUFFER);
		for (int i = 0; i < vertexCount; i++) {
			//convert to our coordinate system:
			vec3 vertex = vertices.at(i);

			//set data
			verticesBuffer[i].x = vertex.x;
			verticesBuffer[i].y = vertex.y;
			verticesBuffer[i].z = vertex.z;
		}
		rtcUnmapBuffer(scene, mesh, RTC_VERTEX_BUFFER);

		///////////////
		//read faces
		///////////////				

		/* set triangles */
		Triangle* triangles = (Triangle*)rtcMapBuffer(scene, mesh, RTC_INDEX_BUFFER);
		for (int i = 0; i < faceCount; i++) {
			//set data: make sure indices start at 0!! 
			Triangle triangle = faces.at(i);

			triangles[i].v0 = triangle.v0;
			triangles[i].v1 = triangle.v1;
			triangles[i].v2 = triangle.v2;

			//triangles[i].v0 = triangle.v0;
			//triangles[i].v1 = triangle.v2;
			//triangles[i].v2 = triangle.v1;

		}
		rtcUnmapBuffer(scene, mesh, RTC_INDEX_BUFFER);

		ObjectData objectData;
		objectData.bsdf = bsdf;
		objectData.name = name;
		addObjectData(mesh, objectData);

		return mesh;
	}

	bool addObjectData(int objectID, const ObjectData& data) {
		if (objectID == (int)objectData.size()) {
			objectData.push_back(data);
			return true;
		}
		else {
			cout << "object id out of range! -> increase MAX_OBJECT_COUNT" << endl;
			return false;
		}
	}

	/* Returns the object data for the given geometryID in the ObjectData struct pointer
	* Return true if obejct was found, false otherwise.
	*/
	inline bool getObjectData(int geometryID, ObjectData* result) {

		if (geometryID >= 0 && geometryID < (int)objectData.size()) {
			*result = objectData[geometryID];
			return true;
		}
		else {
			cout << "ObjectData: index out of bounds!" << endl;
			return false;
		}		
	}

	/* Returns the BSDF for the given geometryID in the ObjectData struct pointer
	* Return true if obejct was found, false otherwise.
	*/
	inline BSDF* getBSDF(int geometryID) {
		if (geometryID >= 0 && geometryID < (int)objectData.size()) {
			return objectData[geometryID].bsdf;
		}
		else {
			cout << "ObjectData: index out of bounds!" << endl;
			return 0;
		}
	}

	/**
	* Shoots a ray specified by its origin and direction through the Scene. Returns true if intersection is found.
	* The struct ray is used to output the intersection information when a hit was found.
	*
	* The intersection normal is stored in the output variable intersectionNormal
	*/
	inline bool intersectScene(const vec3& rayOrigin, const vec3& rayDir, RTCRay& ray, vec3& intersectionNormal)
	{
		ray.time = 0.0f;
		ray.org[0] = rayOrigin.x;
		ray.org[1] = rayOrigin.y;
		ray.org[2] = rayOrigin.z;
		ray.dir[0] = rayDir.x;
		ray.dir[1] = rayDir.y;
		ray.dir[2] = rayDir.z;
		ray.tnear = 0.0f;
		ray.tfar = FLT_MAX;
		ray.geomID = RTC_INVALID_GEOMETRY_ID;
		ray.primID = RTC_INVALID_GEOMETRY_ID;
		ray.mask = -1;

		rtcIntersect(scene, ray);

		bool hit = false;
		if (ray.geomID != RTC_INVALID_GEOMETRY_ID) {
			hit = true;
			vec3 ng = vec3(ray.Ng[0], ray.Ng[1], ray.Ng[2]);
			intersectionNormal = normalize(ng);
		}
		return hit;
	}

	/**
	* Checks for occlusion using a shadowray specified by a vertex and the direction to the light. 
	* Returns true if the vertex is occluded on the segment specified by the direction and distance.
	* The struct ray is used to output the intersection information when a hit was found.
	*/
	inline bool vertexOccluded(const vec3& vertexPosition, const vec3& direction, const float& occlusionDistance, RTCRay& shadowRay)
	{
		shadowRay.org[0] = vertexPosition[0];
		shadowRay.org[1] = vertexPosition[1];
		shadowRay.org[2] = vertexPosition[2];
		shadowRay.dir[0] = direction[0];
		shadowRay.dir[1] = direction[1];
		shadowRay.dir[2] = direction[2];
		//shadowRay.tnear = Constants::epsilon;
		shadowRay.tnear = 0.0f;
		shadowRay.tfar = occlusionDistance - Constants::epsilon;
		shadowRay.geomID = 1;
		shadowRay.primID = 0;
		shadowRay.mask = -1;
		shadowRay.time = 0;

		/* trace shadow ray */
		rtcOccluded(scene, shadowRay);

		return (shadowRay.geomID == 0);
	}

	void readNextLine(ifstream& inputData, StringParser& output) {
		string currentLine;
		StringParser line;
		bool jumpOver = true;

		while (jumpOver) {
			getline(inputData, currentLine);
			line = StringParser(currentLine);
			if (!line.startsWith("#") || currentLine == "") {
				jumpOver = false;
			}
		}
		output = line;
	}

	/**
	* Reads a scene from an xml scene file.
	*/
	bool readSceneXML(const string& settingsFilePath, Rendering* rendering, Medium* medium)
	{
		//check for correct file:
		ifstream file;
		file.open(settingsFilePath, ios::in); // opens as ASCII!
		if (!file) {
			cout << "invalid scene file path: " << settingsFilePath << endl;
			return false;
		}
		file.close();

		//convert xml fle to zero terminated string
		rapidxml::file<> xmlFile(settingsFilePath.c_str());

		if (xmlFile.size() <= 0) {
			cout << "invalid scene file: " << endl;
			return false;
		}

		xml_document<> doc;
		doc.parse<0>(xmlFile.data());

		//read the data:
		xml_node<>* sceneNode = doc.first_node("scene");
		xml_attribute<>* version =  sceneNode->first_attribute("version");
		string v = version->value();
		cout << "Scene: version " << v << endl;
		if (v != "1.4.0") {
			cout << "The scene file version is out of date!" << endl;
			return false;
		}
		else {
			vector<LightSource*> lightSourcesContainer = vector<LightSource*>();

			//Integrator
			xml_node<>* integratorNode = sceneNode->first_node("integrator");
			xml_attribute<>* integratorType = integratorNode->first_attribute("type");
			string integrator = integratorType->value();
			string sessionName;
			int width, height, threadCount, MESC, maxSegments;
			bool renderParallel = true;

			for (xml_node<>* intSubNode = integratorNode->first_node(); intSubNode; intSubNode = intSubNode->next_sibling()) {
				string currName = intSubNode->name();
				if (currName == "output") {
					sessionName = intSubNode->first_attribute("sessionName")->value();
					StringParser widthS = StringParser(intSubNode->first_attribute("width")->value());
					width = widthS.getIntParam("");
					StringParser heightS = StringParser(intSubNode->first_attribute("height")->value());
					height = heightS.getIntParam("");
					if (width < 1 || height < 1) {
						cout << "invalid resolution! " << width << ", " << height << endl;
						return false;
					}
				}
				else if (currName == "rendering") {
					string typeS = intSubNode->first_attribute("type")->value();
					if (typeS == "maxSamples") {
						StringParser maxSampleS = StringParser(intSubNode->first_attribute("spp")->value());
						int samples = maxSampleS.getIntParam("");
						rendering->renderingEnum = RENDER_MAX_SAMPLES;
						rendering->SAMPLE_COUNT = samples;
					}
					else if (typeS == "maxDuration") {
						StringParser maxDurationS = StringParser(intSubNode->first_attribute("maxDuration")->value());
						double maxDur = maxDurationS.getDoubleParam("");
						rendering->renderingEnum = RENDER_MAX_DURATION;
						rendering->MAX_DURATION = maxDur;
					}					
					else {
						cout << "invalid rendering type !" << endl;
						return false;
					}
				}
				else if (currName == "maxPathSegments") {
					StringParser string = StringParser(intSubNode->first_attribute("value")->value());
					maxSegments = string.getIntParam("");
					if (maxSegments < 1) {
						cout << "invalid maxSegments!" << maxSegments << ", value has to be > 0!" << endl;
						return false;
					}
				}
				else if (currName == "MESC") {
					StringParser string = StringParser(intSubNode->first_attribute("value")->value());
					MESC = string.getIntParam("");
					if (MESC < 1) {
						cout << "invalid MESC!" << MESC << ", value has to be > 0!" << endl;
						return false;
					}
				}
				else if (currName == "threads") {
					StringParser string = StringParser(intSubNode->first_attribute("count")->value());
					threadCount = string.getIntParam("");
					if (threadCount == 1) {
						renderParallel = false;
					}
					else if (threadCount < 0) {
						cout << "invalid thread count!"<< threadCount << ", value has to be positive!" << endl;
						return false;
					}
				}
				else if (currName == "lightChoiceStrategy") {
					string strategyString = intSubNode->first_attribute("type")->value();
					if (strategyString == "UNIFORM") {
						lightChoiceStrategy = UNIFORM;
						lightSampler = &Scene::sampleLightPositionUniformly;
						lightVertexPDF = &Scene::getLightPositionSamplingPDFUniform;
					}
					else if (strategyString == "INTENSITY_BASED") {
						lightChoiceStrategy = INTENSITY_BASED;
						lightSampler = &Scene::sampleLightPositionFluxBased;
						lightVertexPDF = &Scene::getLightPositionSamplingPDF_FluxBased;
					}
					else if (strategyString == "INTENSITY_DISTANCE_BASED") {
						lightChoiceStrategy = INTENSITY_DISTANCE_BASED;
						lightSampler = &Scene::sampleLightPositionDistanceAndFluxBased;
						lightVertexPDF = &Scene::getLightPositionSamplingPDF_DistanceAndFluxBased;
					}
					//else if (strategyString == "INTENSITY_DISTANCE_DIRECTION_BASED") {
					//	lightChoiceStrategy = INTENSITY_DISTANCE_DIRECTION_BASED;
					//	lightSampler = &Scene::sampleLightPositionDistanceDirectionFluxBased;
					//	lightVertexPDF = &Scene::getLightPositionSamplingPDF_DistanceDirectionFluxBased;
					//}
					else {
						cout << "invalid light choice strategy!" << endl;
						return false;
					}
				}
				else {
					cout << "invalid integrator param! " << currName << endl;
				}
			}
			rendering->WIDTH = width;
			rendering->HEIGHT = height;
			rendering->MAX_MVNEE_SEGMENTS = MESC;
			rendering->MAX_MVNEE_SEGMENTS_F = (float)MESC;
			rendering->MAX_SEGMENT_COUNT = maxSegments;
			rendering->RENDER_PARALLEL = renderParallel;
			rendering->THREAD_COUNT = threadCount;
			rendering->sessionName = sessionName;
			IntegratorEnum integratorEnum;
			if (integrator == "PATH_TRACING_MVNEE_FINAL") {
				integratorEnum = PATH_TRACING_MVNEE_FINAL;
			}
			else if (integrator == "PATH_TRACING_MVNEE") {
				integratorEnum = PATH_TRACING_MVNEE;
			}
			else if (integrator == "PATH_TRACING_RANDOM_WALK") {
				integratorEnum = PATH_TRACING_RANDOM_WALK;
			}
			else if (integrator == "PATH_TRACING_MVNEE_LIGHT_IMPORTANCE_SAMPLING") {
				integratorEnum = PATH_TRACING_MVNEE_LIGHT_IMPORTANCE_SAMPLING;
			}
			else if (integrator == "PATH_TRACING_MVNEE_LIGHT_IMPORTANCE_SAMPLING_IMPROVED") {
				integratorEnum = PATH_TRACING_MVNEE_LIGHT_IMPORTANCE_SAMPLING_IMPROVED;
			}			
			else  {
				cout << "invalid integrator!! " << integrator << endl;
				return false;
			}
			rendering->integrator = integratorEnum;

			//Camera
			xml_node<>* cameraNode = sceneNode->first_node("camera");
			for (xml_node<>* camSubNode = cameraNode->first_node(); camSubNode; camSubNode = camSubNode->next_sibling()) {
				string currName = camSubNode->name();
				xml_attribute<>* camAttr = camSubNode->first_attribute();
				string camValue = camAttr->value();
				if (currName == "distanceToImagePlane") {
					camera.distanceToImagePlane = (float)atof(camValue.data());
				}
				else if (currName == "imagePlaneWidth") {
					camera.imagePlaneWidth = (float)atof(camValue.data());
				}
				else if (currName == "imagePlaneHeight") {
					camera.imagePlaneHeight = (float)atof(camValue.data());
				}
				else if (currName == "lookAt") {
					StringParser originS = StringParser(camSubNode->first_attribute("origin")->value());
					StringParser targetS = StringParser(camSubNode->first_attribute("target")->value());
					StringParser upS = StringParser(camSubNode->first_attribute("up")->value());

					vec3 origin = originS.getVec3Param("");
					vec3 target = targetS.getVec3Param("");
					vec3 up = normalize(upS.getVec3Param(""));
					vec3 lookAt = normalize(target - origin);
					vec3 right = cross(up, lookAt);

					camera.cameraOrigin = origin;
					camera.camLookAt = lookAt;
					camera.camUp = up;
					camera.camRight = right;
				}
				else {
					cout << "invalid camera param! " << currName << endl;
				}
			}

			//Lightsources

			cout << "adding lights to the scene..." << endl;

			for (xml_node<>* lightNode = sceneNode->first_node("lightsource"); lightNode; lightNode = lightNode->next_sibling("lightsource")) {
				LightTypeEnum lightType;
				string typeString = lightNode->first_attribute("type")->value();
				
				vec3 lightCenter, lightNormal, lightU, lightV, lightColor;
				float lightBrightness, lightRadius;
				double cosExponent;

				for (xml_node<>* lightSubNode = lightNode->first_node(); lightSubNode; lightSubNode = lightSubNode->next_sibling()) {
					string currName = lightSubNode->name();

					if (currName == "position") {
						xml_attribute<>* posAttr = lightSubNode->first_attribute("center");
						StringParser posString = StringParser(posAttr->value());
						lightCenter = posString.getVec3Param("");

						posAttr = lightSubNode->first_attribute("normal");
						StringParser normalString = StringParser(posAttr->value());

						lightNormal = normalize(normalString.getVec3Param(""));

						//create tangent frame from normal direction for u,v directions:
						coordinateSystem(lightNormal, lightU, lightV);
					}
					else if (currName == "radius") {
						xml_attribute<>* radAttr = lightSubNode->first_attribute("value");
						lightRadius = (float)atof(radAttr->value());
					}
					else if (currName == "brightness") {
						xml_attribute<>* brightAttr = lightSubNode->first_attribute("value");
						lightBrightness = (float)atof(brightAttr->value());
					}
					else if (currName == "color") {
						xml_attribute<>* colorAttr = lightSubNode->first_attribute("value");
						StringParser colorString = StringParser(colorAttr->value());
						lightColor = colorString.getVec3Param("");
					}
					else if (currName == "cosExponent") {
						xml_attribute<>* cosExpAttr = lightSubNode->first_attribute("value");
						StringParser cosExpString = StringParser(cosExpAttr->value());
						cosExponent = cosExpString.getDoubleParam("");
					}
					else {
						cout << "invalid light param! " << currName << endl;
					}					
				}

				DiffuseLambertianBSDF* lightBSDF = new DiffuseLambertianBSDF(lightColor);

				if (typeString == "lightdisk") {
					lightType = TypeLightDisk;

					//add light object geometry to embree scene
					int lightID = addCircularPlane(lightCenter, lightNormal, lightU, lightV, lightRadius, 10000, "disklight", lightBSDF);

					LightDisk* currLightSource = new LightDisk(lightCenter, lightNormal, lightU, lightV, lightColor, lightBrightness, lightRadius, lightID);
					//add light source to container
					lightSourcesContainer.push_back(currLightSource);
					
				}
				else if (typeString == "spotlight") {
					lightType = TypeSpotlight;

					//add light object geometry to embree scene
					int lightID = addCircularPlane(lightCenter, lightNormal, lightU, lightV, lightRadius, 10000, "spotlight", lightBSDF);

					SpotLight* currLightSource = new SpotLight(lightCenter, lightNormal, lightU, lightV, lightColor, lightBrightness, lightRadius, cosExponent, lightID);
					//add light source to container
					lightSourcesContainer.push_back(currLightSource);
					
				}
				else {
					cout << "Invalid Light type!" << endl;
					return false;
				}

				
			}

			//create light array and add all lights!
			initializeLights(lightSourcesContainer);	


			//MediumSettings
			xml_node<>* mediumNode = sceneNode->first_node("medium");
			string mediumTypeS = mediumNode->first_attribute("type")->value();
			if (mediumTypeS == "homogeneous") {
				cout << "Reading homogeneous medium..." << endl;

				xml_node<>* coefficientSubNode = mediumNode->first_node("coefficients");
				string mu_s_str = coefficientSubNode->first_attribute("mu_s")->value();
				string mu_a_str = coefficientSubNode->first_attribute("mu_a")->value();

				double mu_s = atof(mu_s_str.data());
				double mu_a = atof(mu_a_str.data());
				double mu_t = mu_s + mu_a;

				double meanFreePath = 1.0 / mu_t;
				float meanFreePathF = (float)meanFreePath;

				xml_node<>* phaseFunctionSubNode = mediumNode->first_node("phaseFunction");
				string g_str = phaseFunctionSubNode->first_attribute("g")->value();
				double g = atof(g_str.data());
				float gF = (float)g;

				medium->hg_g = g;
				medium->hg_g_F = gF;
				medium->meanFreePath = meanFreePath;
				medium->meanFreePathF = meanFreePathF;
				medium->mu_s = mu_s;
				medium->mu_a = mu_a;
				medium->mu_t = mu_t;
				medium->scatteringAlbedo = mu_s / mu_t;
			}
			else {
				cout << "invalid Medium Type!" << endl;
				return false;
			}
						

			//Models/Objects
			cout << "adding models to the scene..." << endl;

			for (xml_node<>* objectNode = sceneNode->first_node("model"); objectNode; objectNode = objectNode->next_sibling("model")) {
				string name = objectNode->first_attribute("name")->value();
				string modelType = objectNode->first_attribute("type")->value();
				unsigned int objID = RTC_INVALID_GEOMETRY_ID;

				if (modelType == "obj") {
					string objFilePath = objectNode->first_node("filename")->first_attribute("value")->value();
					StringParser translationS = StringParser(objectNode->first_node("transform")->first_attribute("translate")->value());
					vec3 translation = translationS.getVec3Param("");
					StringParser scaleS = StringParser(objectNode->first_node("transform")->first_attribute("scale")->value());
					float scale = scaleS.getFloatParam("");
					string flipZS = objectNode->first_node("transform")->first_attribute("flipZ")->value();
					bool flipZ = true;
					if (flipZS == "false") {
						flipZ = false;
					}

					string flipVertexOrderS = objectNode->first_node("transform")->first_attribute("flipVertexOrder")->value();
					bool flipVertexOrder = false;
					if (flipVertexOrderS == "true") {
						flipVertexOrder = true;
					}

					//parse Material:
					BSDF* materialBSDF = 0;

					xml_node<>* materialNode = objectNode->first_node("material");
					string typeS = materialNode->first_attribute("type")->value();

					if (typeS == "diffuse") {
						//Lambertian Diffuse BRDF
						StringParser colorS = StringParser(materialNode->first_attribute("albedo")->value());
						vec3 albedo = colorS.getVec3Param("");
						DiffuseLambertianBSDF* diffuse = new DiffuseLambertianBSDF(albedo);
						materialBSDF = diffuse;
					}					
					else {
						cout << "invalid material type!" << endl;
						return false;
					}					

					objID = addObject(objFilePath, name, materialBSDF, translation, scale, flipZ, flipVertexOrder);
				}
				else if (modelType == "plane") {
					StringParser translationS = StringParser(objectNode->first_node("transform")->first_attribute("y")->value());
					float yTransl = translationS.getFloatParam("");
					StringParser scaleS = StringParser(objectNode->first_node("transform")->first_attribute("scale")->value());
					float scale = scaleS.getFloatParam("");					

					//parse Material:
					BSDF* materialBSDF = 0;

					xml_node<>* materialNode = objectNode->first_node("material");
					string typeS = materialNode->first_attribute("type")->value();

					if (typeS == "diffuse") {
						//Lambertian Diffuse BRDF
						StringParser colorS = StringParser(materialNode->first_attribute("albedo")->value());
						vec3 albedo = colorS.getVec3Param("");
						DiffuseLambertianBSDF* diffuse = new DiffuseLambertianBSDF(albedo);
						materialBSDF = diffuse;
					}					
					else {
						cout << "invalid material type for ground plane!" << endl;
						return false;
					}

					objID = addGroundPlane(name, materialBSDF, scale, yTransl);
				}
				else {
					cout << "invalid model type!" << modelType << endl;
				}

				if (objID == RTC_INVALID_GEOMETRY_ID) {
					cout << "invalid model: object file path incorrect?" << endl;
					return false;
				}
				
			}
		}
		
		cout << "scene xml file successfully parsed." << endl;
		return true;
	}

	void printLightChoiceStrategy(ofstream& o) {
		switch (lightChoiceStrategy) {
			case UNIFORM: o << "UNIFORM" << endl; break;
			case INTENSITY_BASED: o << "INTENSITY_BASED" << endl; break;
			case INTENSITY_DISTANCE_BASED: o << "INTENSITY_DISTANCE_BASED" << endl; break;
			default: cout << "unknown" << endl; break;
		}
	}

	inline bool calcDistancesToLights(float* distanceArray, const vec3& pathVertex) {
		if (lightSourceCount == 1) {
			//distances not relevant, since only one light source is present
			return true;
		}

		for (int i = 0; i < lightSourceCount; i++) {
			float currDist = length(lightSources[i]->center - pathVertex);
			if (isfinite(currDist) && currDist > 0.0f) {
				distanceArray[i] = currDist;
			}
			else {
				cout << "calcDistances: Positions on the light source given, instead of a path vertex!" << endl;
				return false;
			}
		}
		return true;
	}

private:

	/**
	* First samples a light source uniformly. Then a vertex on the chosen light source is sampled.
	* The index of the light source, as well as the sampled vertex are returned.
	*/
	inline vec3 sampleLightPositionUniformly(float* lightDistancesArray, const double& xi1, const double& xi2, const double& xi3, int* lightIndex)
	{
		double lightCount = (double) lightSourceCount;
		double lightPDF = 1.0 / lightCount;
			
		//sample light source
		double currSum = 0.0;
		int finalLightIndex = lightSourceCount - 1;
		for (int i = 0; i < lightSourceCount - 1; i++)
		{
			currSum += lightPDF;
			if (xi1 <= currSum) {
				finalLightIndex = i;
				break;
			}
		}

		vec3 lightPosSample = lightSources[finalLightIndex]->sampleLightPosition(xi2, xi3);

		*lightIndex = finalLightIndex;
		return lightPosSample;
	}

	inline double getLightPositionSamplingPDFUniform(float* lightDistancesArray, int lightIndex)
	{
		double lightCount = (double)lightSourceCount;
		double lightPDF = 1.0 / lightCount;
		double vertexSamplingPDF = lightSources[lightIndex]->getPositionSamplingPDF();

		double samplingPDF = vertexSamplingPDF * lightPDF;
		return samplingPDF;
	}

	/**
	* First samples a light source based on light flux. Then a vertex on the chosen light source is sampled.
	* The index of the light source, as well as the sampled vertex are returned.
	*/
	inline vec3 sampleLightPositionFluxBased(float* lightDistancesArray, const double& xi1, const double& xi2, const double& xi3, int* lightIndex)
	{
		//sample light source
		double currSum = 0.0;
		int finalLightIndex = lightSourceCount - 1;
		for (int i = 0; i < lightSourceCount - 1; i++)
		{
			double currPDF = lightFluxes[i] / lightFluxSum;
			currSum += currPDF;
			if (xi1 <= currSum) {
				finalLightIndex = i;
				break;
			}
		}

		vec3 lightPosSample = lightSources[finalLightIndex]->sampleLightPosition(xi2, xi3);

		*lightIndex = finalLightIndex;
		return lightPosSample;
	}

	inline double getLightPositionSamplingPDF_FluxBased(float* lightDistancesArray, int lightIndex)
	{
		double lightPDF = lightFluxes[lightIndex] / lightFluxSum;;
		double vertexSamplingPDF = lightSources[lightIndex]->getPositionSamplingPDF();

		double samplingPDF = vertexSamplingPDF * lightPDF;
		return samplingPDF;
	}

	/**
	* First samples a light source based on light flux and distance. Then a vertex on the chosen light source is sampled.
	* The index of the light source, as well as the sampled vertex are returned.
	*/
	inline vec3 sampleLightPositionDistanceAndFluxBased(float* lightDistancesArray, const double& xi1, const double& xi2, const double& xi3, int* lightIndex)
	{
		//calculate normalization factor
		double normalizationFactor = 0.0;
		for (int i = 0; i < lightSourceCount; i++)
		{
			double currDistance = (double)lightDistancesArray[i];
			assert(currDistance > 0.0);
			double currValue = (lightFluxes[i] / currDistance);
			assert(isfinite(currValue));
			normalizationFactor += currValue;
		}

		//sample light source
		double currSum = 0.0;
		int finalLightIndex = lightSourceCount - 1;
		for (int i = 0; i < lightSourceCount - 1; i++)
		{
			double currDistance = (double)lightDistancesArray[i];

			double currPDF = (lightFluxes[i] / currDistance) / normalizationFactor;
			currSum += currPDF;
			if (xi1 <= currSum) {
				finalLightIndex = i;
				break;
			}
		}

		vec3 lightPosSample = lightSources[finalLightIndex]->sampleLightPosition(xi2, xi3);

		*lightIndex = finalLightIndex;
		return lightPosSample;
	}

	inline double getLightPositionSamplingPDF_DistanceAndFluxBased(float* lightDistancesArray, int lightIndex)
	{
		double lightValue;

		//calculate normalization factor
		double normalizationFactor = 0.0;
		for (int i = 0; i < lightSourceCount; i++)
		{
			double currDistance = (double)lightDistancesArray[i];
			assert(currDistance > 0.0);
			double currValue = (lightFluxes[i] / currDistance);
			assert(isfinite(currValue));

			if (i == lightIndex) {
				lightValue = currValue;
			}

			normalizationFactor += currValue;
		}


		double lightPDF = lightValue / normalizationFactor;
		double vertexSamplingPDF = lightSources[lightIndex]->getPositionSamplingPDF();

		double samplingPDF = vertexSamplingPDF * lightPDF;
		return samplingPDF;
	}


};
