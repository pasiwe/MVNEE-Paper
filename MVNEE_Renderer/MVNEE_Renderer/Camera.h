#pragma once

#include "glm/glm.hpp"

using glm::vec3;

struct Camera {	
	vec3 cameraOrigin; //focal point, where all camera rays originate from
	vec3 camLookAt; //direction camera looks at
	vec3 camUp; //direction facing up
	vec3 camRight; //last direction forming a left handed coordiante system

	float distanceToImagePlane;

	float imagePlaneWidth;
	float imagePlaneHeight;
};