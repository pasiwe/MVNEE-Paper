#pragma once

#include "glm/glm.hpp"
#include "RenderingUtility.h"
#include "Settings.h"

using glm::vec3;

class LightSource
{
protected:
	//light brightness and color
	vec3 Le;
public:
	vec3 center;

	//left handed coordinate system of the light disk:
	vec3 normal;
	vec3 u;
	vec3 v;
	
	vec3 lightColor;
	float lightBrightness;

	LightTypeEnum lightType;

	int geomID;
public:
	LightSource(const vec3& center, const vec3& normal, const vec3& u, const vec3& v, const vec3& lightColor, const float& lightBrightness, int geomID, LightTypeEnum lightType) :
		center(center),
		normal(normal),
		u(u),
		v(v),
		lightColor(lightColor),
		lightBrightness(lightBrightness),
		Le(lightBrightness * lightColor),
		geomID(geomID),
		lightType(lightType)
	{

	}

	~LightSource() {

	}

	virtual vec3 sampleLightPosition(const double& xi1, const double& xi2) = 0;	
	virtual vec3 sampleLightDirection(const double& xi1, const double& xi2) = 0;

	virtual double getArea() = 0;

	virtual double getPositionSamplingPDF() = 0;
	virtual double getDirectionSamplingPDF(const vec3& sampledDir) = 0;

	/* Checks, if the intersection position lies on the light source */
	virtual bool lightIntersected(int geomID) = 0;

	/* Used to implement normal culling etc. the hit direction is the direction, with which the light was hit */
	virtual bool validHitDirection(const vec3& hitDirection) = 0;

	/* Used to calculate the intensity from the light source.
	* @param hit position on the light source
	* @param inDir direction on the light source
	*/
	virtual double getIntensity(const vec3& position, const vec3& inDir) = 0;

	/* Used to calculate the intensity from the light source multiplied with the light color for the path contribution. 
	* @param hit position on the light source
	* @param inDir direction on the light source
	*/
	virtual vec3 getEmissionIntensity(const vec3& position, const vec3& inDir) = 0;

	/* Returns the type of light */
	LightTypeEnum getType() {
		return lightType;
	}

	virtual void printParameters(ofstream& o) = 0;
	
};

class LightDisk : public LightSource
{
private:
	double lightDiskSamplingPDF;
	double area; 
	const double c_2PI;
public:
	float radius;
	
public:
	LightDisk(const vec3& center, const vec3& normal, const vec3& u, const vec3& v, const vec3& lightColor, const float& lightBrightness, const float& radius, int geomID) :
		LightSource(center, normal, u, v, lightColor, lightBrightness, geomID, TypeLightDisk),
		radius(radius),
		c_2PI(2.0 * M_PI)
	{
		area = (M_PI * (double)radius * (double)radius);
		lightDiskSamplingPDF = (1.0 / area);
	}

	~LightDisk()
	{
	}

	inline vec3 sampleLightPosition(const double& xi1, const double& xi2)
	{
		//sample point on disk:
		float r = radius * (float)(sqrt(xi1));
		assert(isfinite(r));
		float theta = (float)(c_2PI * xi2);
		vec3 lightPosSample = center + u * r * cosf(theta) + v * r * sinf(theta);
		return lightPosSample;
	}

	inline vec3 sampleLightDirection(const double& xi1, const double& xi2)
	{
		float inner = sqrtf(1.0f - (float)xi1);
		float theta = acosf(inner);
		float phi = (float)(c_2PI * xi2);

		vec3 result = sin(theta) * cos(phi) * u + sin(theta) * sin(phi) * v + cos(theta) * normal;
		return result;
	}

	inline double getArea()
	{
		return area;
	}

	inline double getPositionSamplingPDF()
	{
		return lightDiskSamplingPDF;
	}

	inline double getDirectionSamplingPDF(const vec3& sampledDir)
	{
		double cosTheta = (double) dot(sampledDir, normal);
		return cosTheta * M_1_PI; // cosTheta / Pi
	}

	inline bool lightIntersected(int possibleGeomID)
	{
		return geomID == possibleGeomID;
	}

	void printParameters(ofstream& o)
	{
		o << "light type = Disk Area Light" << endl;
		o << "light brightness " << lightBrightness << endl;
		o << "light radius " << radius << endl;
		o << "light center (" << center.x << ", " << center.y << ", " << center.z << ")" << endl;
		o << "light normal (" << normal.x << ", " << normal.y << ", " << normal.z << ")" << endl;
		o << "light u (" << u.x << ", " << u.y << ", " << u.z << ")" << endl;
		o << "light v (" << v.x << ", " << v.y << ", " << v.z << ")" << endl;
	}

	/* Implement normal culling */
	inline bool validHitDirection(const vec3& hitDirection)
	{
		return (dot(-hitDirection, normal) > 0.0f);
	}

	/* Used to calculate the intensity from the light source.
	* @param hit position on the light source
	* @param inDir direction on the light source
	*/
	inline double getIntensity(const vec3& position, const vec3& inDir)
	{
		return (double)lightBrightness;
	}

	/* Used to calculate the intensity from the light source for the path contribution.
	* @param hit position on the light source
	* @param inDir direction on the light source
	*/
	inline vec3 getEmissionIntensity(const vec3& position, const vec3& inDir)
	{
		return Le;
	}
};


class SpotLight : public LightSource
{
private:
	double vertexSamplingPDF;
	double area;
	double cosineExponent;
	double cosineExponent_P_1;

	const double c_2PI;
public:
	float radius;

public:

	SpotLight(const vec3& center, const vec3& normal, const vec3& u, const vec3& v, const vec3& lightColor, const float& lightBrightness, const float& radius, const double& cosineExponent, int geomID) :
		LightSource(center, normal, u, v, lightColor, lightBrightness, geomID, TypeSpotlight),
		radius(radius),
		cosineExponent(cosineExponent),
		cosineExponent_P_1(cosineExponent + 1.0),
		c_2PI(2.0 * M_PI)
	{
		area = (M_PI * (double)radius * (double)radius);
		vertexSamplingPDF = (1.0 / area);
	}

	inline vec3 sampleLightPosition(const double& xi1, const double& xi2)
	{
		//sample point on disk:
		float r = radius * (float)(sqrt(xi1));
		assert(isfinite(r));
		float theta = (float)(c_2PI * xi2);
		vec3 lightPosSample = center + u * r * cosf(theta) + v * r * sinf(theta);
		return lightPosSample;
	}

	inline vec3 sampleLightDirection(const double& xi1, const double& xi2)
	{
		double inner = pow(1.0 - xi1, 1.0 / cosineExponent_P_1);
		double theta = acos(inner);
		double phi = (c_2PI * xi2);

		vec3 result = (float)(sin(theta) * cos(phi)) * u + (float)(sin(theta) * sin(phi)) * v + (float)cos(theta) * normal;
		return result;
	}

	inline double getArea()
	{
		return area;
	}

	inline double getPositionSamplingPDF()
	{
		return vertexSamplingPDF;
	}

	inline double getDirectionSamplingPDF(const vec3& sampledDir)
	{
		double cosTheta = (double)dot(sampledDir, normal);
		double directionPDF = (cosineExponent_P_1)* pow(cosTheta, cosineExponent) / c_2PI;
		return directionPDF; 
	}

	inline bool lightIntersected(int possibleGeomID)
	{
		return possibleGeomID == geomID;
	}

	void printParameters(ofstream& o)
	{
		o << "light type = Spot Light" << endl;
		o << "light brightness " << lightBrightness << endl;
		o << "light radius " << radius << endl;
		o << "light center (" << center.x << ", " << center.y << ", " << center.z << ")" << endl;
		o << "light normal (" << normal.x << ", " << normal.y << ", " << normal.z << ")" << endl;
		o << "light u (" << u.x << ", " << u.y << ", " << u.z << ")" << endl;
		o << "light v (" << v.x << ", " << v.y << ", " << v.z << ")" << endl;
		o << "cosine exponent " << cosineExponent << endl;
	}

	/* Implement normal culling */
	inline bool validHitDirection(const vec3& hitDirection)
	{
		return (dot(-hitDirection, normal) > 0.0f);
	}

	/* Used to calculate the intensity from the light source.
	* @param hit position on the light source
	* @param inDir direction on the light source
	*/
	inline double getIntensity(const vec3& position, const vec3& inDir)
	{
		double cosThetaSpotLight = (double)dot(-inDir, normal);
		assert(isfinite(cosThetaSpotLight));

		/*if (cosThetaSpotLight <= 0.0f) {
			cout << "error: sampled light dir possibly wrong!" << endl;
			return 0.0f;
		}*/
		assert(cosThetaSpotLight > 0.0f);
		double intensity = (cosineExponent_P_1)* pow(cosThetaSpotLight, cosineExponent) / c_2PI;
		return intensity * (double)(lightBrightness);
	}

	/* Used to calculate the intensity from the light source for the path contribution.
	* @param hit position on the light source
	* @param inDir direction on the light source
	*/
	inline vec3 getEmissionIntensity(const vec3& position, const vec3& inDir)
	{
		double intensity = getIntensity(position, inDir);
		return (float)(intensity)* lightColor;
	}
};

