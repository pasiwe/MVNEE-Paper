#pragma once

#include "glm/glm.hpp"
#include <string>

using std::string;
using glm::vec3;

enum LightChoiceStrategy {
	UNIFORM,
	INTENSITY_BASED,
	INTENSITY_DISTANCE_BASED
};

/**
* Enum for all types of lights.
*/
enum LightTypeEnum {
	TypeLightDisk,
	TypeSpotlight
};

enum RenderingEnum {
	RENDER_MAX_SAMPLES,
	RENDER_MAX_DURATION
};


/**
* Enum for the identification of all implemented Integrators
*/
enum IntegratorEnum {
	PATH_TRACING_RANDOM_WALK, //standard path tracing random walk in media
	PATH_TRACING_NEE_MIS, //path tracing + NEE in media
	PATH_TRACING_MVNEE, //base MVNEE version, rather slow
	PATH_TRACING_MVNEE_FINAL, //optimized final MVNEE version
	PATH_TRACING_MVNEE_LIGHT_IMPORTANCE_SAMPLING, //MVNEE adjusted for light source directions
	PATH_TRACING_MVNEE_LIGHT_IMPORTANCE_SAMPLING_IMPROVED, //MVNEE adjusted for light source directions, additionally NEE as third estimator
};

namespace Constants
{
	const float epsilon = 0.0001f;

	//sigma-values for mean cosines 0.0, 0.1,..., 0.9 for 1 mfp:
	const double sigmaPhi[10] = {
		1.07526,
		0.961113,
		0.839835,
		0.716209,
		0.594202,
		0.476602,
		0.365761,
		0.263209,
		0.169391,
		0.0830828
	};

	const double GGX_CONVERSION_Constant = 1.637618734;

	//blockwise mean image brightness
	const int TILES_SIDE = 8;
}

struct Medium
{
	double mu_s; //scattering coefficient
	double mu_a; //absorption coefficient
	double mu_t; //extinction coefficient

	double hg_g; //mean cosine for Henyey Greenstein
	float hg_g_F;

	double scatteringAlbedo;

	double meanFreePath;
	float meanFreePathF;
};

struct Rendering
{
	bool RENDER_PARALLEL;
	int THREAD_COUNT;

	RenderingEnum renderingEnum;

	//maximal path segment count, afterwards rendering is stopped
	int MAX_SEGMENT_COUNT;	

	/* an initial maximum number of MVNEE segments:
	* if the distance to light divided by the mean free path is greater than MAX_MVNEE_SEGMENTS,
	* MVNEE will not be executed!
	*/
	int MAX_MVNEE_SEGMENTS;
	float MAX_MVNEE_SEGMENTS_F;

	//image settings
	int WIDTH;
	int HEIGHT;

	int SAMPLE_COUNT;
	//maximum rendering time, afterwards rendering is stopped
	double MAX_DURATION;

	string sessionName;

	//specify integrator here:
	IntegratorEnum integrator;


};
