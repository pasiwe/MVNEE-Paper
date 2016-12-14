#pragma once
#define _USE_MATH_DEFINES

#include "glm/glm.hpp"
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <string>
#include <iomanip>
#include "Settings.h"

//using namespace glm;
using glm::vec3;
using glm::length;
using glm::normalize;
using namespace std;


enum IntersectionEnum {
	HIT_FROM_BEHIND, //intersection has happened, but from the wrong side
	HIT_FROM_FRONT //intersection has happened from correct side
};

/**
* Intersects a plane with a ray. The plane is defined by its normal and the point p0 that is the closest point to the coordinate system's origin.
* p0 can be calculated given a random point on the plane.
*/
static inline bool rayPlaneIntersect(const vec3& rayOrigin, const vec3& rayDir, const vec3& planeVertex, const vec3&planeNormal, float* t)
{
	float vDotN = dot(rayDir, planeNormal);

	if (vDotN < 0.0f) {
		//correct setting: plane faces in opposite ray direction
		//check if ray and plane are parallel:
		if (fabs(vDotN) <= 0.00001f) {
			return false;
		}

		float shortestDistanceToPlane = dot(planeVertex, planeNormal);
		vec3 p0 = shortestDistanceToPlane * planeNormal;

		vec3 rayOriginToP0 = p0 - rayOrigin;
		float tentative_t = -dot(rayOriginToP0, planeNormal) / fabsf(vDotN);

		if (tentative_t < 0.0f) {
			//ray faces away from the plane and will never hit it
			return false;
		}
		else {
			*t = tentative_t;
			return true;
		}

	}
	else if (vDotN > 0.00001f) {
		//ray hits light from behind

		float shortestDistanceToPlane = dot(planeVertex, planeNormal);
		vec3 p0 = shortestDistanceToPlane * planeNormal;

		vec3 rayOriginToP0 = p0 - rayOrigin;
		float tentative_t = dot(rayOriginToP0, planeNormal) / fabsf(vDotN);

		if (tentative_t < 0.0f) {
			//ray faces away from the plane and will never hit it
			return false;
		}
		else {
			*t = tentative_t;
			return true;
		}
	}
	else {
		//normal culling?
		//cout << "Ray faces in wrong direction!" << endl;
		return false;
	}
}

static inline bool rayCircleIntersect(const vec3& rayOrigin, const vec3& rayDir, const vec3& circleCenter, const vec3& circleNormal, float radius, float* t, IntersectionEnum* isectCase)
{
	bool rayPlaneIntersected = rayPlaneIntersect(rayOrigin, rayDir, circleCenter, circleNormal, t);
	if (rayPlaneIntersected) {
		//check if found intersection lies inside the circle
		float distance = *t;
		vec3 isectPoint = rayOrigin + distance * rayDir;
		float r = length(isectPoint - circleCenter);
		if (r <= radius) {
			//check from which side intersection took place -> NORMAL CULLING!!
			if (dot(rayDir, circleNormal) < 0.0f) {
				*isectCase = HIT_FROM_FRONT;
				return true;
			}
			else {
				*isectCase = HIT_FROM_BEHIND;
				return true;
			}
		}
	}
	return false;
}

/**
* Intersects a plane with a ray. The plane is defined by its normal and the point p0 that is the closest point to the coordinate system's origin.
* p0 can be calculated given a random point on the plane. Specialized version for MVNEE uv-plane!
*/
static inline bool rayPlaneIntersectMVNEE(const vec3& rayOrigin, const vec3& rayDir, const vec3& seedVertex, const vec3&omega, const float& firstSeedDistance, float* t)
{
	float vDotN = dot(rayDir, omega);
	if (vDotN > Constants::epsilon) {
		float tentative_t = firstSeedDistance / vDotN;
		*t = tentative_t;
		return true;
	}
	return false;
}

/**
* Build a tangent space from a given coordinate vector a. The other two coordinate
* vectors b and c are calculated, so a b and c form an orthonormal basis!
*/
static inline void coordinateSystem(const vec3 &a, vec3 &b, vec3 &c) {

	vec3 help;

	if (std::abs(a.x) > std::abs(a.y)) {
		float invLen = 1.0f / sqrtf(a.x * a.x + a.z * a.z);
		help = vec3(a.z * invLen, 0.0f, -a.x * invLen);
		//help = normalize(vec3(a.z * invLen, 0.0f, -a.x * invLen));
	}
	else {
		float invLen = 1.0f / sqrtf(a.y * a.y + a.z * a.z);
		help = vec3(0.0f, a.z * invLen, -a.y * invLen);
		//help = normalize(vec3(0.0f, a.z * invLen, -a.y * invLen));
	}

	//b = normalize(cross(help, a));
	//c = normalize(cross(b, a));
	b = cross(help, a);
	c = help;
}

/**
* Build a tangent space from a given coordinate vector a. The other two coordinate
* vectors b and c are calculated, so a b and c form an orthonormal basis! This function makes sure
* that vectors b and c always lie on the same side of a.
*/
static inline void coordinateSystemAdjusted(const vec3 &a, vec3 &b, vec3 &c) {
	vec3 help;
	bool reversedDir = false;
	if (std::abs(a.x) > std::abs(a.y)) {
		float invLen = 1.0f / std::sqrt(a.x * a.x + a.z * a.z);
		help = normalize(vec3(a.z * invLen, 0.0f, -a.x * invLen));
	}
	else {
		reversedDir = true;
		float invLen = 1.0f / std::sqrt(a.y * a.y + a.z * a.z);
		help = normalize(vec3(0.0f, a.z * invLen, -a.y * invLen));
	}
	if (reversedDir) {
		b = normalize(cross(-help, a));
		c = normalize(cross(-b, a));
		help = b;
		b = c;
		c = help;
	}
	else {
		b = normalize(cross(help, a));
		c = normalize(cross(b, a));
	}
}

/**
* Calculates the sigma of the gauss distribution created by the product of two gaussians:
* N(0, sigma) = N(0, sigma1) * N(0, sigma2)
*/
static inline double calcGaussProduct(const double& sigma1, const double& sigma2) {
	double sigma1_sqr = sigma1 * sigma1;
	double sigma2_sqr = sigma2 * sigma2;

	double factor = (1.0 / sigma1_sqr) + (1.0 / sigma2_sqr);
	double result = sqrt(1.0 / factor);

	return result;
}

/**
* Calculates the sigma of the gauss distribution created by the product of two gaussians:
* In this implementation, the input sigmas are already Expected to be squared!!	
* 
* N(0, sigma) = N(0, sigma1) * N(0, sigma2)
*/
static inline double calcGaussProductSquaredSigmas(const double& sigma1Sqr, const double& sigma2Sqr) {
	//double factor = (1.0 / sigma1Sqr) + (1.0 / sigma2Sqr);
	//double result = sqrt(1.0 / factor);
	double factor = (sigma1Sqr * sigma2Sqr) / (sigma1Sqr + sigma2Sqr);
	double result = sqrt(factor);
	return result;
}

/**
* Calculates the sigma value for the perturbation on a specific vertex of a seedpath. For that the sigma value that discribes one henyey greenstein
* distribution is used.
* @param vertexCount number of vertices of the seedpath, excluding light and fork vertex!!
* @param vertexIndex index out of [0, vertexCount-1] specifying which vertex will be perturbed in the path
* @param hg_sigma sigma value of the gauss distribution which describes the distribution of the henyey-greenstein phase function values over  one vertex hop
*/
static inline double getSigmaForVertex(const int vertexIndex, const int vertexCount, const double& hg_sigma) {
	//computational expensive way:

	int leftFactor = vertexIndex + 1;
	int rightFactor = vertexCount + 1 - leftFactor;

	double sigma1 = (double)(leftFactor)* hg_sigma;
	double sigma2 = (double)(rightFactor)* hg_sigma;


	double sigma_result = calcGaussProduct(sigma1, sigma2);
	//maybe do this?
	//sigma_result = calcGaussProduct(sigma_result, hg_sigma);

	return sigma_result;
}


/**
* Calculates the output of the Henyey-Greenstein phase function for a fiven cosine input and the
* Henyey-Greenstein mean-cosine g
* @param cos_theta cosine of input and output direction as input for phase function
* @param hg_g Henyey-Greenstein mean-cosine parameter
*/
static inline float henyeyGreenstein(const float& cos_theta, const float& hg_g)
{

	if (!(cos_theta >= -1.001f)) {
		cout << "error in henyeyGreensteinPDF: cosTHeta = " << cos_theta << endl;
	}
	if (!(cos_theta <= 1.001f)) {
		cout << "error in henyeyGreensteinPDF: cosTHeta = " << cos_theta << endl;
	}

	assert(cos_theta >= -1.001f);
	assert(cos_theta <= 1.001f);

	assert(isfinite(cos_theta));

	float temp = 1.0f + hg_g*hg_g - 2.0f * hg_g * cos_theta;
	assert(isfinite(temp));
	float result = (1.0f / (float)(4.0 * M_PI)) * (1 - hg_g*hg_g) / (temp * std::sqrt(temp));
	assert(isfinite(result));
	return result;
}

static inline vec3 sampleHenyeyGreensteinDirection(vec3 previousDirection, const double& xi1, const double& xi2, const float& hg_g)
{
	float cosTheta;
	float xi_1 = (float)xi1;
	float xi_2 = (float)xi2;

	if (abs(hg_g) < 0.000001f) {
		cosTheta = 1.0f - 2.0f * xi_1;
	}
	else {
		float sqrTerm = (1.0f - hg_g * hg_g) / (1.0f - hg_g + 2.0f * hg_g * xi_1);
		cosTheta = (1.0f + hg_g * hg_g - sqrTerm * sqrTerm) / (2.0f * hg_g);
	}


	assert(isfinite(cosTheta));

	//sample a direction
	float sinTheta, cosPhi, sinPhi;

	sinTheta = sqrtf(fmaxf(1.0f - cosTheta*cosTheta, 0.0f));
	float angle = 2.0f * (float)M_PI * xi_2;
	cosPhi = cosf(angle);
	sinPhi = sinf(angle);

	//sampled direction in Tangent Space (z-coord is direction of normal)
	vec3 wo = vec3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);

	vec3 tangent, bitangent;
	//build tangent frame:
	coordinateSystem(previousDirection, bitangent, tangent);

	//compute world vector
	vec3 worldSampleDir = wo.x * tangent + wo.y * bitangent + wo.z * previousDirection;
	return normalize(worldSampleDir);
}

static inline double misPowerWeight(double pdfMain, double pdf2)
{
	if (pdfMain + pdf2 == 0.0) {
		return 0.0;
	}
	double p1 = pdfMain * pdfMain;
	double p2 = pdf2 * pdf2;

	return p1 / (p1 + p2);
}

static inline float misBalanceWeight(double mainPDF, double* allPDFs, const int pdfCount)
{
	double denominator = 0.0;
	for (int i = 0; i < pdfCount; i++) {
		double value = allPDFs[i];
		assert(isfinite(value));
		denominator += value;
	}
	assert(isfinite(denominator));

	double weight = mainPDF / denominator;

	if (!isfinite(weight)) {
		//denominator is infinite
		return 0.0f;
	}

	if (weight < 0.0) {
		//probably overflow of some kind
		return 0.0f;
	}

	if (weight > 1.0) {
		cout << "MIS WEIGHT > 1: " << setprecision(12) << weight << endl;
	}

	assert(weight >= 0.0);
	assert(weight <= 1.0);
	return (float)(weight);
}

static inline float misBalanceWeight(const double& mainPDF, const double& pathTracingPDF, double* mvneeEstimatorPDFs, const int mvneePDFCount)
{
	double denominator = pathTracingPDF;
	for (int i = 0; i < mvneePDFCount; i++) {
		double value = mvneeEstimatorPDFs[i];
		assert(isfinite(value));
		denominator += value;
	}
	assert(isfinite(denominator));

	double weight = mainPDF / denominator;

	if (!isfinite(weight)) {
		//denominator is infinite
		return 0.0f;
	}

	if (weight < 0.0) {
		//probably overflow of some kind
		return 0.0f;
	}

	if (weight > 1.0) {
		cout << "MIS WEIGHT > 1: " << setprecision(12) << weight << endl;
		return 0.0f;
	}

	assert(weight >= 0.0);
	assert(weight <= 1.0);
	return (float)(weight);
}

static inline float misBalanceWeight(const double& mainPDF, const double& pathTracingPDF, double* mvneeEstimatorPDFs, const int mvneePDFCount, double* mvneeLISEstimatorPDFs, const int mvneeLISPDFCount)
{
	double denominator = pathTracingPDF;
	for (int i = 0; i < mvneePDFCount; i++) {
		double value = mvneeEstimatorPDFs[i];
		assert(isfinite(value));
		denominator += value;
	}
	assert(isfinite(denominator));

	for (int i = 0; i < mvneeLISPDFCount; i++) {
		double value = mvneeLISEstimatorPDFs[i];
		assert(isfinite(value));
		denominator += value;
	}
	assert(isfinite(denominator));

	double weight = mainPDF / denominator;

	if (!isfinite(weight)) {
		//denominator is infinite
		return 0.0f;
	}

	if (weight < 0.0) {
		//probably overflow of some kind
		return 0.0f;
	}

	if (weight > 1.0) {
		cout << "MIS WEIGHT > 1: " << setprecision(12) << weight << endl;
		return 0.0f;
	}

	assert(weight >= 0.0);
	assert(weight <= 1.0);
	return (float)(weight);
}



static inline float misPowerWeight(const double& mainPDF, double* allPDFs, const int pdfCount)
{
	//find maximum pdf value
	double maxPDFValue = mainPDF;
	bool mainPDFInsideAllPDFs = false;
	for (int i = 0; i < pdfCount; i++) {
		double value = allPDFs[i];
		if (value == mainPDF) {
			mainPDFInsideAllPDFs = true;
		}

		if (!isfinite(value)) {
			cout << "estimatorPDF infinite, return weight 0!!" << endl;
			return 0.0f;
		}
		assert(isfinite(value));
		assert(value >= 0.0);
		maxPDFValue = std::max(value, maxPDFValue);
	}
	//DEBUG:
	//if (!mainPDFInsideAllPDFs) {
	//	return 0.0f;
	//}

	assert(mainPDFInsideAllPDFs);

	double denominator = 0.0;
	for (int i = 0; i < pdfCount; i++) {
		double value = allPDFs[i] / maxPDFValue;
		assert(value >= 0.0);
		double value2 = (value * value);
		assert(isfinite(value2));
		assert(value2 >= 0.0);

		denominator += value2;
	}
	assert(denominator >= 0.0);
	assert(isfinite(denominator));
	if (denominator == 0.0) {
		return 0.0f;
	}
	double numerator = (mainPDF / maxPDFValue) * (mainPDF / maxPDFValue);
	double weight = numerator / denominator;
	if (!isfinite(weight)) {
		//denominator is infinite
		return 0.0f;
	}

	if (weight < 0.0) {
		//probably overflow of some kind
		return 0.0f;
	}

	if (weight > 1.0) {
		cout << "MIS WEIGHT > 1: " << setprecision(12) << weight << endl;
	}

	assert(weight >= 0.0);
	assert(weight <= 1.0);
	return (float)(weight);
}

static inline float misPowerWeight(const double& mainPDF, const double& pathTracingPDF, double* mvneeEstimatorPDFs, const int mvneeEstimatorCount)
{
	//find maximum pdf value
	double maxPDFValue = pathTracingPDF;
	//bool mainPDFInsideAllPDFs = false;
	for (int i = 0; i < mvneeEstimatorCount; i++) {
		double value = mvneeEstimatorPDFs[i];
		//if (value == mainPDF) {
		//	mainPDFInsideAllPDFs = true;
		//}

		if (!isfinite(value)) {
			cout << "estimatorPDF infinite, return weight 0!!" << endl;
			return 0.0f;
		}
		assert(isfinite(value));
		assert(value >= 0.0);
		maxPDFValue = std::max(value, maxPDFValue);
	}
	//DEBUG:
	//if (!mainPDFInsideAllPDFs) {
	//	return 0.0f;
	//}

	//assert(mainPDFInsideAllPDFs);

	double denominator = pathTracingPDF / maxPDFValue;
	denominator *= denominator;
	for (int i = 0; i < mvneeEstimatorCount; i++) {
		double value = mvneeEstimatorPDFs[i] / maxPDFValue;
		assert(value >= 0.0);
		double value2 = (value * value);
		assert(isfinite(value2));
		assert(value2 >= 0.0);

		denominator += value2;
	}
	assert(denominator >= 0.0);
	assert(isfinite(denominator));
	if (denominator == 0.0) {
		return 0.0f;
	}
	double numerator = (mainPDF / maxPDFValue) * (mainPDF / maxPDFValue);
	double weight = numerator / denominator;
	if (!isfinite(weight)) {
		//denominator is infinite
		return 0.0f;
	}

	if (weight < 0.0) {
		//probably overflow of some kind
		return 0.0f;
	}

	if (weight > 1.0) {
		cout << "MIS WEIGHT > 1: " << setprecision(12) << weight << endl;
	}

	assert(weight >= 0.0);
	assert(weight <= 1.0);
	return (float)(weight);
}

/** Samples a free path length in the homogenous medium. Makes sure that the result is > 0, which would
* lead to invalid floating point values later on.
*/
static inline double sampleFreePathLength(const double& xi, const double& mu_t)
{
	//make sure xi != 1 and != 0?
	float freePathLength = 0.0f;
	freePathLength = (float)(-log(1.0 - xi) / mu_t);
	assert(freePathLength >= 0.0f);
	return (double)freePathLength;
}

/** Samples a free path length in the homogenous medium. Makes sure that the result is > 0, which would
* lead to invalid floating point values later on.
*/
static inline float sampleLimitedFreePathLength(const double& xi, const double& mu_t, const float& minDistance, const float& maxDistance)
{
	double xiMin = 1.0 - exp(-mu_t * (double)minDistance);
	double xiMax = 1.0 - exp(-mu_t * (double)maxDistance);

	double width = xiMax - xiMin;

	double xiFinal = xiMin + xi * width;

	float freePathLength = (float)(-log(1.0 - xiFinal) / mu_t);

	assert(freePathLength >= 0.0f);
	return freePathLength;
}

static inline double getLimitedFreePathPDF(const float& sampledDistance ,const double& mu_t, const float& minDistance, const float& maxDistance)
{
	//if (maxDistance <= minDistance) {
	//	cout << "error: limited free path min >= max!" << endl;
	//	return 0.0;
	//}

	assert(maxDistance > minDistance);
	assert(minDistance >= 0.0f);
	assert(sampledDistance > 0.0f);

	double dMinCDF = 1.0 - exp(-mu_t * (double)minDistance);
	double dMaxCDF = 1.0 - exp(-mu_t * (double)maxDistance);

	double normalization = dMaxCDF - dMinCDF;
	assert(isfinite(normalization));
	assert(normalization > 0.0);

	double samplingPDF = mu_t * exp(-mu_t * (double)sampledDistance);

	return samplingPDF / normalization;
}


/**
* Calculates the pdf for sampling a ggx perturbation in 2D. For this pdf, only the square of the perturbation radius is necessary as input, as
* the direction is uniform.
* The pdf is the product of sampling a uniform angle [0,2PI], the sampling of radius using the positive ggx and the normalization factor.
* @param r_square SQUARE of the radius
*/
static inline double GGX_2D_PDF(const float& r_square, const double& alpha)
{
	double alpha2 = alpha * alpha;
	double factor = alpha2 + (double)r_square;
	double pdf = (alpha2) / (M_PI * factor * factor);
	return pdf;
}

/**
* Samples a 2D vector of a 2D GGX Distribution. Formula is retrieved by inverse CDF
* of the GGX 2D PDF in Polar coordinates.
* @param xi1: uniform random variable in [0,1)
* @param xi2: uniform random variable in [0,1)
*/
static inline glm::vec2 sampleGGX2D(const double& s, const double& xi1, const double& xi2)
{
	assert(xi1 < 1.0);
	double theta = 2.0 * M_PI * xi2;

	double r = sqrt((s*s*xi1) / (1.0 - xi1));
	assert(isfinite(r));

	double x = r * sin(theta);
	double y = r * cos(theta);
	glm::vec2 result((float)x, (float)y);
	return result;
}

static inline bool isNotZero(const vec3& vector) {
	if (vector.x != 0.0f || vector.y != 0.0f || vector.z != 0.0f) {
		return true;
	}
	return false;
}

static inline bool isZero(const vec3& vector) {
	if (vector.x == 0.0f && vector.y == 0.0f && vector.z == 0.0f) {
		return true;
	}
	return false;
}