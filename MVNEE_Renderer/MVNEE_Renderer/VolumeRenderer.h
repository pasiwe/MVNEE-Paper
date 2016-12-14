#pragma once

//Compilerflags: 
#define NDEBUG


#include "Settings.h"
#include "RenderingUtility.h"
#include "Scene.h"
#include "Path.h"
#include <omp.h>
#include <random>
#include <chrono>
#include <algorithm>

using namespace std;
using glm::vec3;


/**
* This class contains the main rendering code for all different integrators, as well as several special sampling techniques.
*
*
*/
class VolumeRenderer
{
private:
	RTCScene sceneGeometry;
	Scene* scene;
	Medium medium;
	Rendering rendering;

	vec3* frameBuffer;

	double sigmaForHG;

	//variables for time measurement:
	std::chrono::system_clock::time_point measureStart;
	std::chrono::system_clock::time_point measureStop;


	//random generator, one for every thread (maximum 16 threads?)
	std::random_device rd;
	//mersenne twister random generator
	std::mt19937* mt;
	//for equally distributed random values between 0.0 and 1.0
	std::uniform_real_distribution<double>* distribution;


	//Every thread has its own Path:
	Path** pathTracingPaths;

	//seed and perturbed vertex arrays pre-initialized
	vec3** seedVertices;
	vec3** perturbedVertices;
	//arrays for the pdfs of all estimators
	double** estimatorPDFs;
	//arrays for the pdfs of all estimators
	double** estimatorPDFsLIS;
	//pdfs for path tracing up to a specific vertex position
	double** cumulatedPathTracingPDFs;
	//arrays for the seed path segment lengths
	float** seedSegmentLengths;
	double** seedSegmentLengthSquares;

	float** distancesToLightSources;
	

	/*
	* Integrator function pointer: the integrator function is chosen at run time.
	* All Integrators have to return a vec3 and have two vec3 parameters, the ray origin and
	* starting direction.
	*/
	vec3 (VolumeRenderer::*integrator)(const vec3& rayOrigin, const vec3& rayDir) = NULL;

public:
	VolumeRenderer(RTCScene sceneGeometry, Scene* scene, Medium& medium, Rendering& rendering);
	~VolumeRenderer();

	/**
	* Renders the scene and stops once the specified spp count is reached.
	*/
	void renderScene();

	/**
	* Renders the scene and stops once the maximum duration is exceeded.
	*/
	void renderSceneWithMaximalDuration(double maxDurationMinutes);


	void writeBufferToFloatFile(const string& fileName, int width, int height, vec3* buffer);
	vec3* readFloatFileToBuffer(const string& fileName, int* width, int* height);

	//writing to output image:
	void saveBufferToTGA(const char* filename, vec3* imageBuffer, int imageWidth, int imageHeight);
	

private:

	/* Path Tracing with medium interaction */
	vec3 pathTracingRandomWalk(const vec3& rayOrigin, const vec3& rayDir);

	/* Path Tracing and NEE with medium interaction*/
	vec3 pathTracing_NEE_MIS(const vec3& rayOrigin, const vec3& rayDir);
	

	/**
	* Calculates the measurement contribution as well as the path tracing PDF of the given path.
	* On top of that, the PDFs of all MVNEE estimators are calculated.
	*
	* As a result, the contribution of the path is calculated using the pdf of the estimator that created the path originally.
	* This contribution is weighted by the MIS weight using the estimator PDFs.
	*
	* For this method, the light is expected to be an area light, also all objects in the scene are expected to have a diffuse lambertian BRDF.
	*
	* @param path containing all vertices of the path from camera to light source, as well as surface information (normal objectID,etc.)
	* @param estimatorIndex index of the estimator that created the path:
	*			estimatorIndex = 0: path tracing
	*			estimatorIndex > 0: MVNEE starting after i path tracing segments
	* @param lightSourceIndex index of the light source, that this path ends on
	* @return: the MIS weighted contribution of the path
	*/
	inline vec3 calcFinalWeightedContribution(Path* path, int estimatorIndex, int lightSourceIndex);


	/**
	* Combination of path tracing with Multiple Vertex Next Event Estimation (MVNEE) for direct lighting calculation at vertices in the medium,
	* as well as on surfaces. Creates one path tracing path and multiple MVNEE pathsstarting at the given rayOrigin with the given direction.
	* This function returns the MIS weighted summed contributions of all created paths from the given origin to the light source.
	*
	* The light in this integrator is expected to be an area light, also all objects in the scene are expected to have a diffuse lambertian BRDF.
	*
	* As MVNEE seed paths, a line connection is used. Seed distances are sampled using transmittance distance sampling.
	* MVNEE perturbation is performed using GGX2D sampling in the u-v-plane of the seed vertices.
	*/
	vec3 pathTracing_MVNEE(const vec3& rayOrigin, const vec3& rayDir);

	/**
	* Calculates the measurement contribution as well as the path tracing PDF of the given path.
	* On top of that, the PDFs of all MVNEE estimators are calculated.
	*
	* As a result, the contribution of the path is calculated using the pdf of the estimator that created the path originally.
	* This contribution is weighted by the MIS weight using the estimator PDFs.
	*
	* For this method, the light is expected to be an area light, also all objects in the scene are expected to have a diffuse lambertian BRDF.
	*
	* @param path containing all vertices of the path from camera to light source, as well as surface information (normal objectID,etc.)
	* @param estimatorIndex index of the estimator that created the path:
	*			estimatorIndex = 0: path tracing
	*			estimatorIndex > 0: MVNEE starting after i path tracing segments
	* @param lightSourceIndex index of the light source, that this path ends on
	* @return: the MIS weighted contribution of the path
	*/
	inline vec3 calcFinalWeightedContribution_FINAL(Path* path, int estimatorIndex, int lightSourceIndex, const int firstPossibleMVNEEEstimatorIndex, const double& currentMeasurementContrib, const vec3& currentColorThroughput);


	/**
	* Combination of path tracing with Multiple Vertex Next Event Estimation (MVNEE) for direct lighting calculation at vertices in the medium,
	* as well as on surfaces. Creates one path tracing path and multiple MVNEE pathsstarting at the given rayOrigin with the given direction.
	* This function returns the MIS weighted summed contributions of all created paths from the given origin to the light source.
	*
	* The light in this integrator is expected to be an area light, also all objects in the scene are expected to have a diffuse lambertian BRDF.
	*
	* As MVNEE seed paths, a line connection is used. Seed distances are sampled using transmittance distance sampling.
	* MVNEE perturbation is performed using GGX2D sampling in the u-v-plane of the seed vertices.
	*/
	vec3 pathTracing_MVNEE_FINAL(const vec3& rayOrigin, const vec3& rayDir);

	

	/**
	* Calculates the measurement contribution as well as the path tracing PDF of the given path.
	* On top of that, the PDFs of all MVNEE estimators are calculated. THIS VERSION samples one segment from the light source and attempts a connection to this new vertex.
	*
	* As a result, the contribution of the path is calculated using the pdf of the estimator that created the path originally.
	* This contribution is weighted by the MIS weight using the estimator PDFs.
	*
	* For this method, the light is expected to be an area light, also all objects in the scene are expected to have a diffuse lambertian BRDF.
	*
	* @param path containing all vertices of the path from camera to light source, as well as surface information (normal objectID,etc.)
	* @param estimatorIndex index of the estimator that created the path:
	*			estimatorIndex = 0: path tracing
	*			estimatorIndex > 0: MVNEE starting after i path tracing segments
	* @param lightSourceIndex index of the light source, that this path ends on
	* @return: the MIS weighted contribution of the path
	*/
	inline vec3 calcFinalWeightedContribution_LightImportanceSampling(Path* path, int estimatorIndex, int lightSourceIndex, const int firstPossibleMVNEEEstimatorIndex, const double& currentMeasurementContrib, const vec3& currentColorThroughput);


	/**
	* Combination of path tracing with Multiple Vertex Next Event Estimation (MVNEE) for direct lighting calculation at vertices in the medium,
	* as well as on surfaces. Creates one path tracing path and multiple MVNEE pathsstarting at the given rayOrigin with the given direction.
	* This function returns the MIS weighted summed contributions of all created paths from the given origin to the light source.
	*
	* THIS VERSION samples one segment from the light source and attempts a connection to this new vertex.
	*
	* The light in this integrator is expected to be an area light, also all objects in the scene are expected to have a diffuse lambertian BRDF.
	*
	* As MVNEE seed paths, a line connection is used. Seed distances are sampled using transmittance distance sampling.
	* MVNEE perturbation is performed using GGX2D sampling in the u-v-plane of the seed vertices.
	*/
	vec3 pathTracing_MVNEE_LightImportanceSampling(const vec3& rayOrigin, const vec3& rayDir);


	/**
	* Calculates the measurement contribution as well as the path tracing PDF of the given path.
	* On top of that, the PDFs of all MVNEE estimators are calculated. THIS VERSION samples one segment from the light source and attempts a connection to this new vertex.
	* An extra one-segment-connection handling is provided.
	*
	* As a result, the contribution of the path is calculated using the pdf of the estimator that created the path originally.
	* This contribution is weighted by the MIS weight using the estimator PDFs.
	*
	* For this method, the light is expected to be an area light, also all objects in the scene are expected to have a diffuse lambertian BRDF.
	*
	* @param path containing all vertices of the path from camera to light source, as well as surface information (normal objectID,etc.)
	* @param estimatorIndex index of the estimator that created the path:
	*			estimatorIndex = 0: path tracing
	*			estimatorIndex > 0: MVNEE starting after i path tracing segments
	* @param lightSourceIndex index of the light source, that this path ends on
	* @return: the MIS weighted contribution of the path
	*/
	inline vec3 calcFinalWeightedContribution_LightImportanceSamplingImproved(Path* path, int estimatorIndex, int lightSourceIndex, const int firstPossibleMVNEEEstimatorIndex, const double& currentMeasurementContrib, const vec3& currentColorThroughput);


	/**
	* Combination of path tracing with Multiple Vertex Next Event Estimation (MVNEE) for direct lighting calculation at vertices in the medium,
	* as well as on surfaces. Creates one path tracing path and multiple MVNEE pathsstarting at the given rayOrigin with the given direction.
	* This function returns the MIS weighted summed contributions of all created paths from the given origin to the light source.
	*
	* THIS VERSION samples one segment from the light source and attempts a connection to this new vertex. An extra one-segment-connection handling is provided.
	*
	* The light in this integrator is expected to be an area light, also all objects in the scene are expected to have a diffuse lambertian BRDF.
	*
	* As MVNEE seed paths, a line connection is used. Seed distances are sampled using transmittance distance sampling.
	* MVNEE perturbation is performed using GGX2D sampling in the u-v-plane of the seed vertices.
	*/
	vec3 pathTracing_MVNEE_LightImportanceSamplingImproved(const vec3& rayOrigin, const vec3& rayDir);

	/**
	* Constructs a MVNEE connection between start and end vertex. All vertices but the start vertex are added to the path object. Visibility checks are performed.
	* Light normal culling is not done, since the end vertex might not be a light vertex. In case creation is not possible , false is returned, otherwise true.
	*/
	bool constructMVNEEConnection(const PathVertex& startVertex, const vec3& endVertex, Path* path, const int threadID, int* mvneeSegmentCount);


	inline double sample1D(const int threadID);
	inline double sample1DOpenInterval(const int threadID);

	inline double gaussPDF(const float& x, const double& stdDeviation);


	/**
	* Samples the appropriate segment count. This is done by summing up sampled free path lengths (transmittance formula as in path tracing)
	* until the curve_length is exceeded. Also returns the segments in an array (maximal MAX_PATH_LENGTH).
	* Return value is false if MAX_SEGMENT_LENGTH is exceeded, true otherwise.
	*/
	inline bool sampleSeedSegmentLengths(const double& curve_length, float* segments, int* segmentCount, const int threadID);

	/**
	* Samples the appropriate segment count. This is done by summing up sampled free path lengths (transmittance formula as in path tracing)
	* until the curve_length is exceeded. Also returns the segments in an array (maximal MAX_PATH_LENGTH) as well as the squared segment lengths.
	* Return value is false if MAX_SEGMENT_LENGTH is exceeded, true otherwise.
	*/
	inline bool sampleSeedSegmentLengths(const double& curve_length, float* segments, double* segmentSquares, int* segmentCount, const int threadID);

	
	/**
	* Samples the appropriate segment count. This is done by summing up sampled free path lengths (transmittance formula as in path tracing)
	* until the curve_length is exceeded. Also returns the segments in an array (maximal MAX_PATH_LENGTH) as well as the squared segment lengths.
	* The muT factor is given as parameter!
	* Return value is false if MAX_SEGMENT_LENGTH is exceeded, true otherwise.
	*/
	inline bool sampleSeedSegmentLengths(const double& curve_length, double muT, float* segments, double* segmentSquares, int* segmentCount, const int threadID);

	
	/** Perturb a vertex on the tangent plane using a ggx sampled radius, and a uniformly sampled angle
	* @param input: vector that has to be perturbed
	* @param alpha width of ggx bell curve
	* @param u: first tangent vector as perturbation direction 1
	* @param v: second tangent vector as perturbation direction 2
	* @return output: perturbed input vector
	*/
	inline void perturbVertexGGX2D(const double& alpha, const vec3& u, const vec3& v, const vec3& input, vec3* output, const int threadID);	


	//output meta files:
	void printRenderingParameters(int sampleCount, double duration);
	vec3 calcMeanImageBrightnessBlockwise();

	
};
