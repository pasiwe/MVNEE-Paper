#include "VolumeRenderer.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include <iostream>
#include <glm/glm.hpp>
#include <omp.h>
#include <embree2/rtcore.h>
#include <embree2/rtcore_ray.h>
#include <embree2/rtcore_geometry.h>
#include <embree2/rtcore_scene.h>
#include "Scene.h"
#include "Settings.h"

using namespace std;
using glm::vec3;


int main(int argc, const char* argv[])
{

	cout << "-----------------------" << endl;
	cout << "VolumeRenderer started." << endl;
	cout << "-----------------------" << endl << endl;

	//tests for compability:
	RTCDevice device = rtcNewDevice(NULL);
	RTCScene scene = rtcDeviceNewScene(device, RTC_SCENE_STATIC, RTC_INTERSECT1);

	//create and fill scene
	Scene* sceneObject = new Scene(scene);

	Medium mediumSetting;
	Rendering renderingSetting;
	bool xmlParsingSuccessful = false;

	if (argc == 1) {
		cout << "No scene file provided: Default Settings used." << endl;
		xmlParsingSuccessful = sceneObject->readSceneXML("../setups/default.xml", &renderingSetting, &mediumSetting);
	}
	else if (argc == 2) {
		//scene file potentially provided
		string settingsFilePath = argv[1];
		xmlParsingSuccessful = sceneObject->readSceneXML(settingsFilePath, &renderingSetting, &mediumSetting);
	}

	if (!xmlParsingSuccessful) {
		cout << "invalid object file path: exit program" << endl;
		rtcDeleteScene(scene);
		rtcDeleteDevice(device);

		cout << "exit program by typing a letter..." << endl;
		string s;
		cin >> s;

		exit(EXIT_FAILURE);
	}

	//commit after all scene geometry is finished
	rtcCommit(scene);

	//create volumeRenderer and start rendering
	VolumeRenderer* volumeRenderer = new VolumeRenderer(scene, sceneObject, mediumSetting, renderingSetting);

	cout << "----------------------" << endl;
	cout << "Start rendering..." << endl;

	if (renderingSetting.renderingEnum == RENDER_MAX_SAMPLES) {
		volumeRenderer->renderScene();
	}
	else if (renderingSetting.renderingEnum == RENDER_MAX_DURATION) {
		volumeRenderer->renderSceneWithMaximalDuration(renderingSetting.MAX_DURATION);
	}
	else {
		cout << "invalid rendering enum!" << endl;
	}

	delete volumeRenderer;

	rtcDeleteScene(scene);
	rtcDeleteDevice(device);
}