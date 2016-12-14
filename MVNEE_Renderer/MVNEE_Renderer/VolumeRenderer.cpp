#include "VolumeRenderer.h"

VolumeRenderer::VolumeRenderer(RTCScene sceneGeometry, Scene* scene, Medium& medium, Rendering& rendering) :
	sceneGeometry(sceneGeometry),
	scene(scene),
	medium(medium),
	rendering(rendering)
{
	
	//if sequential execution wished, set omp thread count to 1
	if (rendering.RENDER_PARALLEL) {
		//assert(omp_get_max_threads() >= rendering.THREAD_COUNT);
		if (omp_get_max_threads() < rendering.THREAD_COUNT) {
			cout << "specified THREAD_COUNT = " << rendering.THREAD_COUNT << " exceeds maximum value of the machine: " << omp_get_max_threads() << endl;
			cout << "please change thread count to a value in range [1, " << omp_get_max_threads() << "]" << endl;
			cout << "exit program by typing a letter..." << endl;
			string s;
			cin >> s;
			exit(EXIT_FAILURE);
		}

	}


	const int threadCount = rendering.THREAD_COUNT;
	//initialize arrays
	mt = new std::mt19937[threadCount];
	distribution = new std::uniform_real_distribution<double>[threadCount];
	pathTracingPaths = new Path*[threadCount];
	seedVertices = new vec3*[threadCount];
	perturbedVertices = new vec3*[threadCount];
	estimatorPDFs = new double*[threadCount];
	estimatorPDFsLIS = new double*[threadCount];
	cumulatedPathTracingPDFs = new double*[threadCount];
	seedSegmentLengths = new float*[threadCount];
	seedSegmentLengthSquares = new double*[threadCount];
	distancesToLightSources = new float*[threadCount];

	//initialize per thread data
	for (int i = 0; i < threadCount; i++) {
		mt[i] = std::mt19937(rd());
		distribution[i] = std::uniform_real_distribution<double>(0.0, 1.0);

		//arrays:
		seedVertices[i] = new vec3[rendering.MAX_SEGMENT_COUNT];
		perturbedVertices[i] = new vec3[rendering.MAX_SEGMENT_COUNT];
		seedSegmentLengths[i] = new float[rendering.MAX_SEGMENT_COUNT];
		seedSegmentLengthSquares[i] = new double[rendering.MAX_SEGMENT_COUNT];
		estimatorPDFs[i] = new double[rendering.MAX_SEGMENT_COUNT];
		estimatorPDFsLIS[i] = new double[rendering.MAX_SEGMENT_COUNT];
		cumulatedPathTracingPDFs[i] = new double[rendering.MAX_SEGMENT_COUNT + 1];

		pathTracingPaths[i] = new Path(rendering.MAX_SEGMENT_COUNT);

		distancesToLightSources[i] = new float[scene->lightSourceCount];
	}
	frameBuffer = new vec3[rendering.HEIGHT * rendering.WIDTH];

	//given the mean cosine, calculate the appropriate sigmaForHg_scaledForMFP:
	float t = medium.hg_g_F * 10.0f;
	int lower = (int)floorf(t);
	int upper = (int)ceilf(t);
	double misWeight = t - floorf(t);
	//linear interpolation of sigmaPhi if between two values:
	if (upper <= 9) {
		sigmaForHG = mix(Constants::sigmaPhi[lower], Constants::sigmaPhi[upper], misWeight);
	}
	else {
		sigmaForHG = mix(Constants::sigmaPhi[lower], 0.0, misWeight);
	}

}

VolumeRenderer::~VolumeRenderer()
{

	delete[] frameBuffer;
	delete scene;

	//delete per thread data
	for (int i = 0; i < rendering.THREAD_COUNT; i++) {
		//delete arrays:
		Path* pathPointer = pathTracingPaths[i];
		delete pathPointer;

		delete[] cumulatedPathTracingPDFs[i];
		delete[] seedVertices[i];
		delete[] perturbedVertices[i];
		delete[] seedSegmentLengths[i];
		delete[] seedSegmentLengthSquares[i];
		delete[] estimatorPDFs[i];		
		delete[] estimatorPDFsLIS[i];
		delete[] distancesToLightSources[i];
	}

	//delete arrays:
	delete[] mt;
	delete[] distribution;
	delete[] pathTracingPaths;
	delete[] seedVertices;
	delete[] perturbedVertices;
	delete[] estimatorPDFs;
	delete[] estimatorPDFsLIS;
	delete[] cumulatedPathTracingPDFs;
	delete[] seedSegmentLengths;
	delete[] seedSegmentLengthSquares;
	delete[] distancesToLightSources;
}

void VolumeRenderer::renderScene()
{
	//if sequential execution wished, set omp thread count to 1
	if (rendering.RENDER_PARALLEL) {
		//assert(omp_get_max_threads() >= rendering.THREAD_COUNT);
		if (omp_get_max_threads() >= rendering.THREAD_COUNT) {
			omp_set_num_threads(rendering.THREAD_COUNT);
			cout << "max num omp threads = " << omp_get_max_threads() << endl;
		}
		else {
			cout << "specified THREAD_COUNT = " << rendering.THREAD_COUNT << " exceeds maximum value of the machine: " << omp_get_max_threads() << endl;
			cout << "please change THREAD_COUNT in Settings.h to a value in range [1, " << omp_get_max_threads() << "]" << endl;
			cout << "exit program by typing a letter..." << endl;
			string s;
			cin >> s;
			exit(EXIT_FAILURE);
		}

	}
	else {
		cout << "sequential execution" << endl;
		omp_set_num_threads(1);
	}


	//choose the integrator:
	switch (rendering.integrator) {
		case PATH_TRACING_RANDOM_WALK: integrator = &VolumeRenderer::pathTracingRandomWalk; break;
		case PATH_TRACING_NEE_MIS: integrator = &VolumeRenderer::pathTracing_NEE_MIS; break;
		case PATH_TRACING_MVNEE: integrator = &VolumeRenderer::pathTracing_MVNEE; break;
		case PATH_TRACING_MVNEE_FINAL: integrator = &VolumeRenderer::pathTracing_MVNEE_FINAL; break;
		case PATH_TRACING_MVNEE_LIGHT_IMPORTANCE_SAMPLING: integrator = &VolumeRenderer::pathTracing_MVNEE_LightImportanceSampling; break;
		case PATH_TRACING_MVNEE_LIGHT_IMPORTANCE_SAMPLING_IMPROVED: integrator = &VolumeRenderer::pathTracing_MVNEE_LightImportanceSamplingImproved; break;
	}

	//Start time measurement
	measureStart = std::chrono::system_clock::now();


	float pixelWidth = scene->camera.imagePlaneWidth / (float)rendering.WIDTH;
	float pixelHeight = scene->camera.imagePlaneHeight / (float)rendering.HEIGHT;
	vec3 pixelStartPos = scene->camera.cameraOrigin - ((scene->camera.imagePlaneWidth + pixelWidth) / 2.0f) * scene->camera.camRight - ((scene->camera.imagePlaneHeight + pixelHeight) / 2.0f) * scene->camera.camUp + scene->camera.distanceToImagePlane * scene->camera.camLookAt;
	
	const float num_samples = (float)rendering.SAMPLE_COUNT;
	
	//shoot ray through every pixel
#if defined _WIN32
	#pragma omp parallel for schedule(dynamic)
#else
	#pragma omp parallel for schedule(dynamic) collapse(2)	
#endif
	for (int j = 0; j < rendering.HEIGHT; j++) {
		for (int i = 0; i < rendering.WIDTH; i++) {
			int threadID = omp_get_thread_num();
			int frameBufferIndex;

			frameBufferIndex = i + j * rendering.WIDTH;
			vec3 pixelContribution = vec3(0.0f);

			//pivot position for subpixel sampling
			vec3 currentPixelOrigin = pixelStartPos + (float)i *pixelWidth * scene->camera.camRight + (float)j * pixelHeight * scene->camera.camUp;

			//multiple random walks per pixel:
			for (int sample = 0; sample < rendering.SAMPLE_COUNT; sample++) {
				//sample position within pixel rectangle:
				float xDiff = (float)sample1D(threadID) * pixelWidth;
				float yDiff = (float)sample1D(threadID) * pixelHeight;

				vec3 rayOrigin = currentPixelOrigin + xDiff * scene->camera.camRight + yDiff * scene->camera.camUp;
				vec3 rayDir = normalize(rayOrigin - scene->camera.cameraOrigin);

				//call integrator function:
				pixelContribution += (*this.*integrator)(rayOrigin, rayDir);

			}

			pixelContribution /= num_samples;

			//contribute to pixel
			frameBuffer[frameBufferIndex] = pixelContribution;			
		}
#if defined _WIN32
		cout << "row " << j << " finished" << endl;
#endif
	}

	//Stop time measurement
	measureStop = std::chrono::system_clock::now();

	auto durationV = std::chrono::duration_cast<std::chrono::microseconds>(measureStop - measureStart).count();
	double duration = (double)durationV / (60.0 * 1000000.0);

	printRenderingParameters(rendering.SAMPLE_COUNT, duration);

	writeBufferToFloatFile(rendering.sessionName, rendering.WIDTH, rendering.HEIGHT, frameBuffer);

	string tgaFileName = rendering.sessionName;
	tgaFileName.append(".tga");
	saveBufferToTGA(tgaFileName.c_str(), frameBuffer, rendering.WIDTH, rendering.HEIGHT);	

	vec3 mib = calcMeanImageBrightnessBlockwise();
}


/**
* Renders the scene and stops once the maximum duration is exceeded.
*/
void VolumeRenderer::renderSceneWithMaximalDuration(double maxDurationMinutes)
{
	//if sequential execution wished, set omp thread count to 1
	if (rendering.RENDER_PARALLEL) {
		//assert(omp_get_max_threads() >= rendering.THREAD_COUNT);
		if (omp_get_max_threads() >= rendering.THREAD_COUNT) {
			omp_set_num_threads(rendering.THREAD_COUNT);
			cout << "max num omp threads = " << omp_get_max_threads() << endl;
		}
		else {
			cout << "specified THREAD_COUNT = " << rendering.THREAD_COUNT << " exceeds maximum value of the machine: " << omp_get_max_threads() << endl;
			cout << "please change THREAD_COUNT in Settings.h to a value in range [1, " << omp_get_max_threads() << "]" << endl;
			cout << "exit program by typing a letter..." << endl;
			string s;
			cin >> s;
			exit(EXIT_FAILURE);
		}

	}
	else {
		cout << "sequential execution" << endl;
		omp_set_num_threads(1);
	}


	//choose the integrator:
	switch (rendering.integrator) {
		case PATH_TRACING_RANDOM_WALK: integrator = &VolumeRenderer::pathTracingRandomWalk; break;
		case PATH_TRACING_NEE_MIS: integrator = &VolumeRenderer::pathTracing_NEE_MIS; break;
		case PATH_TRACING_MVNEE: integrator = &VolumeRenderer::pathTracing_MVNEE; break;
		case PATH_TRACING_MVNEE_FINAL: integrator = &VolumeRenderer::pathTracing_MVNEE_FINAL; break;
		case PATH_TRACING_MVNEE_LIGHT_IMPORTANCE_SAMPLING: integrator = &VolumeRenderer::pathTracing_MVNEE_LightImportanceSampling; break;
		case PATH_TRACING_MVNEE_LIGHT_IMPORTANCE_SAMPLING_IMPROVED: integrator = &VolumeRenderer::pathTracing_MVNEE_LightImportanceSamplingImproved; break;
	}	

	float pixelWidth = scene->camera.imagePlaneWidth / (float)rendering.WIDTH;
	float pixelHeight = scene->camera.imagePlaneHeight / (float)rendering.HEIGHT;
	vec3 pixelStartPos = scene->camera.cameraOrigin - ((scene->camera.imagePlaneWidth + pixelWidth) / 2.0f) * scene->camera.camRight - ((scene->camera.imagePlaneHeight + pixelHeight) / 2.0f) * scene->camera.camUp + scene->camera.distanceToImagePlane * scene->camera.camLookAt;

	
	//shoot ray through every pixel
	int samples = 0;
	bool timeNotElapsed = true;
	double currentDuration = 0.0;

	while (timeNotElapsed) {
		samples++;

		//Start time measurement
		measureStart = std::chrono::system_clock::now();

		//#pragma omp parallel for schedule(dynamic) collapse(2)
		#pragma omp parallel for schedule(dynamic)
		for (int j = 0; j < rendering.HEIGHT; j++) {
			for (int i = 0; i < rendering.WIDTH; i++) {
				int threadID = omp_get_thread_num();
				int frameBufferIndex;

				frameBufferIndex = i + j * rendering.WIDTH;

				//pivot position for subpixel sampling
				vec3 currentPixelOrigin = pixelStartPos + (float)i *pixelWidth * scene->camera.camRight + (float)j * pixelHeight * scene->camera.camUp;
				
				//sample position within pixel rectangle:
				float xDiff = (float)sample1D(threadID) * pixelWidth;
				float yDiff = (float)sample1D(threadID) * pixelHeight;

				vec3 rayOrigin = currentPixelOrigin + xDiff * scene->camera.camRight + yDiff * scene->camera.camUp;
				vec3 rayDir = normalize(rayOrigin - scene->camera.cameraOrigin);

				//call integrator function:
				vec3 pixelContribution = (*this.*integrator)(rayOrigin, rayDir);

				//contribute to pixel
				frameBuffer[frameBufferIndex] += pixelContribution;
			}			
		}
		//Stop time measurement
		measureStop = std::chrono::system_clock::now();				

		//calc elapsed time
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(measureStop - measureStart).count();
		double frameDuration = (double)duration / (60.0 * 1000000.0);
		currentDuration += frameDuration;

		if (samples % 100 == 0) {
			cout << "frame " << samples << " finished." << endl;
			cout << "elapsed time = " << currentDuration << endl;
		}

		//if maxTime exceeded, stop rendering!
		if (currentDuration >= maxDurationMinutes) {
			timeNotElapsed = false;
		}
	}

	int frameBufferIndex;
	for (int j = 0; j < rendering.HEIGHT; j++) {
		for (int i = 0; i < rendering.WIDTH; i++) {
			frameBufferIndex = i + j * rendering.WIDTH;
			frameBuffer[frameBufferIndex] /= (float)samples;
		}
	}	

	printRenderingParameters(samples, currentDuration);	

	writeBufferToFloatFile(rendering.sessionName, rendering.WIDTH, rendering.HEIGHT, frameBuffer);

	string tgaFileName = rendering.sessionName;
	tgaFileName.append(".tga");
	saveBufferToTGA(tgaFileName.c_str(), frameBuffer, rendering.WIDTH, rendering.HEIGHT);

	vec3 mib = calcMeanImageBrightnessBlockwise();
}

vec3 VolumeRenderer::calcMeanImageBrightnessBlockwise()
{
	vec3 meanImageBrightness = vec3(0.0f);

	vec3 meanBrightnessBlocks[Constants::TILES_SIDE][Constants::TILES_SIDE];
	int blockWidth = rendering.WIDTH / Constants::TILES_SIDE;
	int blockHeight = rendering.HEIGHT / Constants::TILES_SIDE;	

	int lastBlockWidth = blockWidth + rendering.WIDTH % Constants::TILES_SIDE;
	int lastBlockHeight = blockHeight + rendering.HEIGHT % Constants::TILES_SIDE;

	//calculate mean image brightness per block
	for (int y = 0; y < Constants::TILES_SIDE; y++) {
		for (int x = 0; x < Constants::TILES_SIDE; x++) {
			vec3 block_mib = vec3(0.0f);
			int startX = x * blockWidth;
			int startY = y * blockHeight;

			int currWidth = blockWidth;
			int currHeight = blockHeight;

			if (x == Constants::TILES_SIDE - 1) {
				currWidth = lastBlockWidth;
			}
			if (y == Constants::TILES_SIDE - 1) {
				currHeight = lastBlockHeight;
			}

			float numPixelsPerBlock = (float)(currWidth*currHeight);

			for (int j = 0; j < currHeight; j++) {
				for (int i = 0; i < currWidth; i++) {
					int widthIndex = startX + i;
					/*if (widthIndex >= rendering.WIDTH) {
						cout << "width index out of bounds!" << endl;
					}*/
					int heightIndex = startY + j;
					/*if (heightIndex >= rendering.HEIGHT) {
						cout << "height index out of bounds!" << endl;
					}*/
					int index = widthIndex + heightIndex * rendering.WIDTH;
					vec3 currentPixel = frameBuffer[index];
					block_mib += currentPixel;
					meanImageBrightness += currentPixel;
				}
			}
			meanBrightnessBlocks[x][y] = block_mib / numPixelsPerBlock;
		}
	}

	//entire image brightness
	meanImageBrightness = meanImageBrightness / ((float)(rendering.WIDTH) * (float)(rendering.HEIGHT));

	//save textual output
	string blockMIBFileRED = rendering.sessionName;
	blockMIBFileRED.append("_RED.mib");
	ofstream oR(blockMIBFileRED.c_str(), ios::out);
	for (int j = 0; j < Constants::TILES_SIDE; j++) {
		for (int i = 0; i < Constants::TILES_SIDE; i++) {
			oR << meanBrightnessBlocks[i][j].r << ", ";
		}
		oR << endl;
	}
	oR << endl;
	oR << "whole image mib: " << meanImageBrightness.r << endl;
	//close the file
	oR.close();

	string blockMIBFileGREEN = rendering.sessionName;
	blockMIBFileGREEN.append("_GREEN.mib");
	ofstream oG(blockMIBFileGREEN.c_str(), ios::out);
	for (int j = 0; j < Constants::TILES_SIDE; j++) {
		for (int i = 0; i < Constants::TILES_SIDE; i++) {
			oG << meanBrightnessBlocks[i][j].g << ", ";
		}
		oG << endl;
	}
	oG << endl;
	oG << "whole image mib: " << meanImageBrightness.g << endl;
	//close the file
	oG.close();

	string blockMIBFileBLUE = rendering.sessionName;
	blockMIBFileBLUE.append("_BLUE.mib");
	ofstream oB(blockMIBFileBLUE.c_str(), ios::out);
	for (int j = 0; j < Constants::TILES_SIDE; j++) {
		for (int i = 0; i < Constants::TILES_SIDE; i++) {
			oB << meanBrightnessBlocks[i][j].b << ", ";
		}
		oB << endl;
	}
	oB << endl;
	oB << "whole image mib: " << meanImageBrightness.b << endl;
	//close the file
	oB.close();


	return meanImageBrightness;
}

double VolumeRenderer::sample1D(const int threadID)
{
	double xi = distribution[threadID](mt[threadID]);
	//while(xi == 1.0) {
	//	xi = distribution[threadID](mt[threadID]);
	//}
	return xi;
}

double VolumeRenderer::sample1DOpenInterval(const int threadID)
{
	double xi = distribution[threadID](mt[threadID]);
	while (xi == 1.0) { //|| xi == 0.0) {
		xi = distribution[threadID](mt[threadID]);
	}
	return xi;
}



vec3 VolumeRenderer::pathTracingRandomWalk(const vec3& rayOrigin, const vec3& rayDir)
{
	int threadID = omp_get_thread_num();
	vec3 finalValue(0.0f);
	vec3 colorThroughput(1.0f);
	double measurementContrib = 1.0;
	double pdf = 1.0;

	vec3 currPosition = rayOrigin;
	vec3 currDir = rayDir;
	int currSegmentLength = 0;

	while (currSegmentLength < rendering.MAX_SEGMENT_COUNT && isNotZero(colorThroughput)) {

		//sample free path Lenth
		double freePathLength = sampleFreePathLength(sample1DOpenInterval(threadID), medium.mu_t);

		//intersect scene
		RTCRay ray;
		vec3 intersectionNormal;
		if (scene->intersectScene(currPosition, currDir, ray, intersectionNormal) && ray.tfar <= freePathLength) {
			if (ray.tfar > Constants::epsilon) {				

				//surface interaction
				vec3 intersectionPos = currPosition + ray.tfar * currDir;

				//check if light was hit:
				int hitLightSourceIndex;
				if (scene->lightIntersected(ray.geomID, &hitLightSourceIndex)) {
					LightSource* hitLight = scene->lightSources[hitLightSourceIndex];
					float cosThetaLight = dot(-currDir, hitLight->normal);

					//Normal culling at light source
					if (hitLight->validHitDirection(currDir)) {
						double transmittance = exp(-medium.mu_t * ray.tfar);
						double GLight = (double)cosThetaLight / ((double)ray.tfar * (double)ray.tfar);
						assert(isfinite(GLight));

						//update mc and pdf:
						measurementContrib *= transmittance * GLight;
						pdf *= transmittance* GLight;
						assert(isfinite(measurementContrib));
						assert(isfinite(pdf));
						double finalContribution = measurementContrib / pdf;
						assert(isfinite(finalContribution));

						finalValue = (float)finalContribution * colorThroughput * hitLight->getEmissionIntensity(intersectionPos, currDir);
					}
					break;
				}


				//surface normal culling:
				if (dot(intersectionNormal, -currDir) <= 0.0f) {
					break;
				}

				/////////////////////////
				// Sample BRDF
				/////////////////////////
				//diffuse surface was hit: get objectData
				BSDF* bsdfData = scene->getBSDF(ray.geomID);

				//update mc and pdf:
				double transmittance = exp(-medium.mu_t * ray.tfar);
				//sample new direction:
				vec3 newDir;
				if (bsdfData->isDelta()) {
					//sample new direction:
					newDir = bsdfData->sampleBSDFDirection(intersectionNormal, currDir, sample1D(threadID), sample1D(threadID));
					measurementContrib *= transmittance;
					pdf *= transmittance;
				}
				else {
					//sample new direction:
					newDir = bsdfData->sampleBSDFDirection(intersectionNormal, currDir, sample1D(threadID), sample1D(threadID));
					if (!bsdfData->validOutputDirection(intersectionNormal, newDir)) {
						break;
					}

					//update measurementContrib and pdf:	
					measurementContrib *= transmittance;
					colorThroughput *= bsdfData->evalBSDF(intersectionNormal, newDir, currDir);
					pdf *= transmittance * bsdfData->getBSDFDirectionPDF(intersectionNormal, newDir, currDir);
				}

				currPosition = intersectionPos + Constants::epsilon * intersectionNormal;
				currDir = newDir;
			}
			else {
				break;
			}
		}
		else {
			//medium interaction
			vec3 nextPosition = currPosition + (float)freePathLength * currDir;

			//sample new direction
			vec3 newDir = sampleHenyeyGreensteinDirection(currDir, sample1D(threadID), sample1D(threadID), medium.hg_g_F);

			float cos_theta = dot(currDir, newDir);
			float phase = henyeyGreenstein(cos_theta, medium.hg_g_F);
			double transmittance = exp(-medium.mu_t * freePathLength);
			double G = 1.0 / (freePathLength * freePathLength);
			if (!isfinite(G)) {
				break;
			}
			assert(isfinite(G));

			//update mc and pdf:
			measurementContrib *= medium.mu_s * (double)phase * transmittance * G;
			pdf *= medium.mu_t * (double)phase * transmittance * G;

			currPosition = nextPosition;
			currDir = newDir;
		}

		currSegmentLength++;
	}
	return finalValue;
}

vec3 VolumeRenderer::pathTracing_NEE_MIS(const vec3& rayOrigin, const vec3& rayDir)
{
	vec3 finalValue(0.0f);
	vec3 colorThroughput(1.0f);
	double measurementContrib = 1.0;
	double pdf = 1.0;

	vec3 currPosition = rayOrigin;
	vec3 currDir = rayDir;

	bool lastVertexOnSurface = false;
	BSDF* lastBSDF = 0;
	double lastDirSamplingPDF = 1.0;
	int threadID = omp_get_thread_num();

	float* tl_LightDistances = distancesToLightSources[threadID];

	int currSegmentLength = 0;

	while (currSegmentLength < rendering.MAX_SEGMENT_COUNT && isNotZero(colorThroughput)) {

		//sample free path Lenth
		double freePathLength = sampleFreePathLength(sample1DOpenInterval(threadID), medium.mu_t);

		//intersect scene
		RTCRay ray;
		vec3 intersectionNormal;
		if (scene->intersectScene(currPosition, currDir, ray, intersectionNormal) && ray.tfar <= freePathLength) {
			if (ray.tfar > Constants::epsilon) {

				//surface interaction
				vec3 intersectionPos = currPosition + ray.tfar * currDir;

				//check if light was hit:
				int hitLightIndex;
				if (scene->lightIntersected(ray.geomID, &hitLightIndex)) {
					LightSource* hitLightSource = scene->lightSources[hitLightIndex];
					float cosThetaLight = dot(-currDir, hitLightSource->normal);

					//Normal culling at light source
					if (hitLightSource->validHitDirection(currDir)) {
						double transmittance = exp(-medium.mu_t * ray.tfar);
						double GLight = (double)cosThetaLight / ((double)ray.tfar * (double)ray.tfar);
						assert(isfinite((float)GLight));

						//update mc and pdf:
						measurementContrib *= transmittance * GLight;
						pdf *= transmittance* GLight;
						assert(isfinite(measurementContrib));
						assert(isfinite(pdf));
						double finalContribution = measurementContrib / pdf;
						assert(isfinite(finalContribution));

						//contribute distance sampling!
						double pathTracingPDF = lastDirSamplingPDF * GLight * transmittance;
						double misWeight = 1.0;
						if (currSegmentLength > 0.0) {
							if (!lastVertexOnSurface || (lastBSDF != 0 && !lastBSDF->isDelta())) {
								bool validLightDistances = scene->calcDistancesToLights(tl_LightDistances, currPosition);
								assert(validLightDistances);

								double lightVertexSamplingPDF = scene->getLightVertexSamplingPDF(tl_LightDistances, hitLightIndex);
								misWeight = misPowerWeight(pathTracingPDF, lightVertexSamplingPDF);
							}
						}

						finalValue += (float)(misWeight * finalContribution) * colorThroughput * hitLightSource->getEmissionIntensity(intersectionPos, currDir);
					}
					break;
				}
				lastVertexOnSurface = true;

				//surface normal culling:
				float cosThetaSurface = dot(intersectionNormal, -currDir);
				if (cosThetaSurface <= 0.0f) {
					break;
				}

				double G = (double)cosThetaSurface / ((double)ray.tfar * (double)ray.tfar);

				currSegmentLength++;
				if (currSegmentLength >= rendering.MAX_SEGMENT_COUNT) {
					break;
				}

				//diffuse surface was hit: get objectData
				BSDF* bsdfData = scene->getBSDF(ray.geomID);
				lastBSDF = bsdfData;

				//update transmittance up to current position
				double transmittance = exp(-medium.mu_t * ray.tfar);
				measurementContrib *= transmittance * G;
				pdf *= transmittance * G;

				/////////////////////////
				// NEE 
				/////////////////////////
				if (!bsdfData->isDelta()) {
					int sampledLightIndex;

					bool validLightDistances = scene->calcDistancesToLights(tl_LightDistances, intersectionPos);
					assert(validLightDistances);

					vec3 lightSample = scene->sampleLightPosition(tl_LightDistances, sample1D(threadID), sample1D(threadID), sample1D(threadID), &sampledLightIndex);
					if (sampledLightIndex > -1) {
						LightSource* sampledLight = scene->lightSources[sampledLightIndex];
						vec3 isecToLight = lightSample - intersectionPos;
						float distToLight = length(isecToLight);
						vec3 dirToLight = isecToLight / distToLight;

						//normal culling at surface
						if (bsdfData->validOutputDirection(intersectionNormal, dirToLight) && distToLight > Constants::epsilon) {
							RTCRay shadowRay;
							if (!scene->vertexOccluded(intersectionPos + Constants::epsilon * intersectionNormal, dirToLight, distToLight - Constants::epsilon, shadowRay)) {
								double pathTracingPDF = bsdfData->getBSDFDirectionPDF(intersectionNormal, dirToLight, currDir);
								float cosThetaLight = dot(-dirToLight, sampledLight->normal);
								//normal culling at light
								if (sampledLight->validHitDirection(dirToLight)) {
									double GLight = (double)cosThetaLight / ((double)distToLight * (double)distToLight);
									double neeTransmittance = exp(-medium.mu_t * (double)distToLight);

									//contribute distance sampling!
									pathTracingPDF *= GLight * neeTransmittance;

									double lightVertexSamplingPDF = scene->getLightVertexSamplingPDF(tl_LightDistances, sampledLightIndex);
									double misWeight = misPowerWeight(lightVertexSamplingPDF, pathTracingPDF);

									vec3 neeColorThroughput = colorThroughput * bsdfData->evalBSDF(intersectionNormal, dirToLight, currDir);
									double neeMeasurementContrib = measurementContrib * neeTransmittance * GLight;
									double neePDF = pdf * lightVertexSamplingPDF;

									double finalContribution = misWeight * (neeMeasurementContrib / neePDF);
									assert(isfinite(finalContribution));

									finalValue += (float)finalContribution * neeColorThroughput * sampledLight->getEmissionIntensity(lightSample, dirToLight);
								}
							}
						}
					}
				}
				/////////////////////////
				// Sample BRDF
				/////////////////////////				

				//update mc and pdf:
				//sample new direction:
				vec3 newDir = bsdfData->sampleBSDFDirection(intersectionNormal, currDir, sample1D(threadID), sample1D(threadID));
				if (!bsdfData->isDelta()) {
					if (!bsdfData->validOutputDirection(intersectionNormal, newDir)) {
						break;
					}

					//update PDF and measurement contrib:
					lastDirSamplingPDF = bsdfData->getBSDFDirectionPDF(intersectionNormal, newDir, currDir);

					if (lastDirSamplingPDF == 0.0) {
						//invalid sampling!
						break;
					}

					pdf *= lastDirSamplingPDF;
					colorThroughput *= bsdfData->evalBSDF(intersectionNormal, newDir, currDir);
				}

				currPosition = intersectionPos + Constants::epsilon * intersectionNormal;
				currDir = newDir;
			}
			else {
				break;
			}
		}
		else {
			//medium interaction
			lastVertexOnSurface = false;

			vec3 nextPosition = currPosition + (float)freePathLength * currDir;

			currSegmentLength++;
			if (currSegmentLength >= rendering.MAX_SEGMENT_COUNT) {
				break;
			}

			//update transmittance up to current position
			double G = 1.0 / (freePathLength * freePathLength);
			if (!isfinite(G)) {
				break;
			}
			assert(isfinite(G));
			double transmittance = exp(-medium.mu_t * freePathLength);
			measurementContrib *= transmittance * G;
			pdf *= medium.mu_t * transmittance * G;

			/////////////////////////
			// NEE in Medium
			/////////////////////////
			int sampledLightIndex;
			bool validLightDistances = scene->calcDistancesToLights(tl_LightDistances, nextPosition);
			assert(validLightDistances);

			vec3 lightSample = scene->sampleLightPosition(tl_LightDistances, sample1D(threadID), sample1D(threadID), sample1D(threadID), &sampledLightIndex);
			if (sampledLightIndex > -1) {
				LightSource* sampledLightSource = scene->lightSources[sampledLightIndex];
				vec3 posToLight = lightSample - nextPosition;
				float distToLight = length(posToLight);
				vec3 dirToLight = posToLight / distToLight;

				if (distToLight > Constants::epsilon) {
					RTCRay shadowRay;
					//check visibility:
					if (!scene->vertexOccluded(nextPosition, dirToLight, distToLight, shadowRay)) {

						float cosTheta = dot(currDir, dirToLight);
						double neePhase = (double)henyeyGreenstein(cosTheta, medium.hg_g_F);
						double pathTracingPDF = neePhase;
						float cosThetaLight = dot(-dirToLight, sampledLightSource->normal);
						//normal culling at light
						if (sampledLightSource->validHitDirection(dirToLight)) {
							double GLight = cosThetaLight / ((double)distToLight * (double)distToLight);
							double neeTransmittance = exp(-medium.mu_t * (double)distToLight);

							//contribute transmittance:
							pathTracingPDF *= GLight * neeTransmittance;

							double lightVertexSamplingPDF = scene->getLightVertexSamplingPDF(tl_LightDistances, sampledLightIndex);

							double misWeight = misPowerWeight(lightVertexSamplingPDF, pathTracingPDF);

							double neeMeasurementContrib = measurementContrib * medium.mu_s * neePhase * neeTransmittance * GLight;
							double neePDF = pdf * lightVertexSamplingPDF;
							assert(isfinite(neePDF));
							assert(neePDF > 0.0f);

							double finalContribution = misWeight * (neeMeasurementContrib / neePDF);
							assert(isfinite(finalContribution));

							finalValue += (float)finalContribution * colorThroughput * sampledLightSource->getEmissionIntensity(lightSample, dirToLight);
						}
					}
				}
			}
			/////////////////////////
			// sample new direction
			/////////////////////////
			vec3 newDir = sampleHenyeyGreensteinDirection(currDir, sample1D(threadID), sample1D(threadID), medium.hg_g_F);

			float cos_theta = dot(currDir, newDir);
			float phase = henyeyGreenstein(cos_theta, medium.hg_g_F);
			lastDirSamplingPDF = (double)phase;

			//update mc and pdf:
			measurementContrib *= medium.mu_s * (double)phase;
			pdf *= (double)phase;

			currPosition = nextPosition;
			currDir = newDir;
		}
	}
	return finalValue;
}

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
vec3 VolumeRenderer::calcFinalWeightedContribution(Path* path, int estimatorIndex, int lightSourceIndex)
{
	const int segmentCount = path->getSegmentLength();
	assert(path->getVertexCount() >= 2);
	assert(segmentCount <= rendering.MAX_SEGMENT_COUNT);
	assert(estimatorIndex >= 0);
	assert(estimatorIndex < segmentCount);
	vec3 errorValue(0.0f); //returned in case of errors

	LightSource* hitLightSource = scene->lightSources[lightSourceIndex];

	//get thread local pre-reservers data for this thread:
	int threadID = omp_get_thread_num();
	vec3* tl_seedVertices = seedVertices[threadID];
	double* tl_cumulatedPathTracingPDFs = cumulatedPathTracingPDFs[threadID];
	float* tl_seedSegmentLengths = seedSegmentLengths[threadID];
	double* tl_estimatorPDFs = estimatorPDFs[threadID];
	float* tl_lightDistances = distancesToLightSources[threadID];

	double measurementContrib = 1.0;
	double pathTracingPDF = 1.0;

	vec3 lastDirection;
	vec3 lightVertex = path->getVertexPosition(segmentCount);

	vec3 colorThroughput = vec3(1.0f, 1.0f, 1.0f);

	tl_cumulatedPathTracingPDFs[0] = 1.0;

	/////////////////////////////////////////////////////////////
	// Find the suffix of the path, that could have been created with MVNEE
	/////////////////////////////////////////////////////////////
	int firstPossibleMVNEEEstimatorIndex = 1;
	//start at vertex before the light vertex and find last surface vertex:
	for (int i = segmentCount - 1; i > 0; i--) {
		PathVertex v;
		path->getVertex(i, v);
		if (v.vertexType == TYPE_SURFACE) {
			BSDF* bsdf = scene->getBSDF(v.geometryID);
			if (bsdf->isDelta()) {
				firstPossibleMVNEEEstimatorIndex = i + 1;
			}
			else {
				firstPossibleMVNEEEstimatorIndex = i;
			}
			break;
		}
	}

	/////////////////////////////////////////////////////////////
	// calculate measurementContribution as well as path tracing pdfs
	/////////////////////////////////////////////////////////////

	//first segment needs special handling, as no phase function is calculated:
	PathVertex firstVertex, secondVertex;
	path->getVertex(0, firstVertex);
	assert(firstVertex.vertexType == TYPE_ORIGIN);
	path->getVertex(1, secondVertex);
	assert(secondVertex.vertexType != TYPE_MVNEE);

	vec3 startDir = secondVertex.vertex - firstVertex.vertex;
	float firstDistance = length(startDir);
	if (firstDistance <= 0.0f) {
		return errorValue;
	}
	startDir = startDir / firstDistance; //normalize startDir

	//special case: light was immediately hit after 1 segment:
	if (path->getVertexCount() == 2) {		
		//Normal culling
		if (hitLightSource->validHitDirection(startDir)) {
			return hitLightSource->getEmissionIntensity(secondVertex.vertex, startDir);
		} 
		return errorValue;		
	}
	
	double firstTransmittance = exp(-medium.mu_t * (double)firstDistance);
	double firstG = 1.0 / ((double)(firstDistance)* (double)(firstDistance));
	assert(isfinite(firstG));
	if (secondVertex.vertexType == TYPE_SURFACE) {
		float cosTheta = dot(secondVertex.surfaceNormal, -startDir);
		if (cosTheta <= 0.0f) {
			cout << "Surface cosTheta <= 0.0f: " << setprecision(12) << cosTheta << endl;
			return errorValue;
		}
		assert(cosTheta > 0.0f);
		firstG *= (double)cosTheta;

		measurementContrib *= firstTransmittance * firstG;
		pathTracingPDF *= firstTransmittance * firstG;
	}
	else {
		measurementContrib *= firstTransmittance * firstG;
		pathTracingPDF *= medium.mu_t * firstTransmittance * firstG;
	}

	//first path tracing pdf calculated:
	tl_cumulatedPathTracingPDFs[1] = pathTracingPDF;

	//calc measurement contrib and pdf for all remaining vertices:
	vec3 previousDir = startDir;
	for (int i = 2; i < path->getVertexCount(); i++) {
		PathVertex currVert, prevVert;
		path->getVertex(i, currVert);
		path->getVertex(i - 1, prevVert);
		vec3 currentDir = currVert.vertex - prevVert.vertex;
		float currentDistance = length(currentDir);

		if (currentDistance <= 0.0f) {			
			return errorValue;
		}
		currentDir = currentDir / currentDistance; //normalize direction

		double currTransmittance = exp(-medium.mu_t * (double)currentDistance);
		double currG = 1.0 / ((double)(currentDistance) * (double)(currentDistance));		
		assert(isfinite(currG));

		//contribute phase function * mu_s / BRDF depending on vertex
		if (prevVert.vertexType == TYPE_SURFACE) {
			//BRDF direction

			BSDF* bsdfData = scene->getBSDF(prevVert.geometryID);

			if (!bsdfData->validOutputDirection(prevVert.surfaceNormal, currentDir)) {
				cout << "wrong outgoing direction at surface!" << endl; 
				return errorValue;
			}

			if (!bsdfData->isDelta()) {
				colorThroughput *= bsdfData->evalBSDF(prevVert.surfaceNormal, currentDir, previousDir);
				pathTracingPDF *= bsdfData->getBSDFDirectionPDF(prevVert.surfaceNormal, currentDir, previousDir);
			}
		}
		else {
			//phase direction
			float cosThetaPhase = dot(previousDir, currentDir);
			double phase = (double)henyeyGreenstein(cosThetaPhase, medium.hg_g_F);

			measurementContrib *= medium.mu_s * phase;
			pathTracingPDF *= phase;
		}
				
		if (i == segmentCount) {
			//Light connection segment:
			if (hitLightSource->validHitDirection(currentDir)) {
				lastDirection = currentDir;

				//special treatment when light source was hit:
				float cosLightF = dot(-currentDir, hitLightSource->normal);
				assert(cosLightF >= 0.0f);
				currG *= (double)cosLightF;

				measurementContrib *= currTransmittance * currG;
				pathTracingPDF *= currTransmittance * currG;
			}
			else {
				cout << "path hits light from backside!" << endl;
				return errorValue;
			}
		}
		//contribute G-Terms, transmittance: take care of medium coefficients!
		else {
			if (currVert.vertexType == TYPE_SURFACE) {
				float cosTheta = dot(currVert.surfaceNormal, -currentDir);
				if (cosTheta <= 0.0f) {
					//cout << "Surface cosTheta <= 0.0f: " << setprecision(12) << cosTheta<< endl;
					return errorValue;
				}
				assert(cosTheta > 0.0f);
				currG *= (double)cosTheta;

				measurementContrib *= currTransmittance * currG;
				pathTracingPDF *= currTransmittance * currG;
			}
			else {
				measurementContrib *= currTransmittance * currG;
				pathTracingPDF *= medium.mu_t * currTransmittance * currG;
			}			
		}
		
		tl_cumulatedPathTracingPDFs[i] = pathTracingPDF;
		previousDir = currentDir;
	}
	assert(isfinite(pathTracingPDF));

	//set path tracing pdf at index 0
	tl_estimatorPDFs[0] = pathTracingPDF;	

	//set estimator pdfs to 0 for all cases where estimator can not have created this path!
	for (int i = 1; i < firstPossibleMVNEEEstimatorIndex; i++) {
		tl_estimatorPDFs[i] = 0.0;
	}

	//////////////////////////////////////
	//calculate estimator PDFs:
	//////////////////////////////////////
	for (int e = firstPossibleMVNEEEstimatorIndex; e < segmentCount; e++) {
		double pathTracingStartPDF = tl_cumulatedPathTracingPDFs[e]; //pdf for sampling the first vertices with path tracing
		vec3 forkVertex = path->getVertexPosition(e); //last path tracing vertex and anchor point for the mvnee path		

		vec3 forkToLight = lightVertex - forkVertex;
		float distanceToLight = length(forkToLight);

		float expectedSegments = distanceToLight / medium.meanFreePathF;

		if (distanceToLight <= 0.0f) {
			tl_estimatorPDFs[e] = 0.0;
		}
		else if (isfinite(expectedSegments) && expectedSegments > 0.0f && expectedSegments <= rendering.MAX_MVNEE_SEGMENTS_F) {
			//MVNEE is only valid if expected segment count is smaller than MAX_MVNEE_SEGMENTS!			
			vec3 omega = forkToLight / distanceToLight; //normalize direction

			//segment count of mvnee path, which is also the mvnee vertex count
			int mvneeSegmentCount = segmentCount - e;

			//build tangent frame: (same for every seed vertex)
			vec3 u, v;
			coordinateSystem(omega, u, v);

			/////////////////////////////////
			// recover seed distances and seed vertices:
			/////////////////////////////////
			bool validMVNEEPath = true;
			float lastSeedDistance = 0.0f;
			for (int s = 0; s < (mvneeSegmentCount - 1); s++) {
				vec3 perturbedVertex = path->getVertexPosition(e + 1 + s);
				vec3 forkToPerturbed = perturbedVertex - forkVertex;

				//use orthogonal projection to recover the seed distance:
				float seedDistance = dot(forkToPerturbed, omega);

				//now some things have to be checked in order to make sure the path can be created with mvnee:
				//the new Distance from forkVertex has to be larger than the summed distance,
				//also the summedDistance may never be larger than distanceToLight!!
				if (seedDistance <= 0.0f || seedDistance <= lastSeedDistance || seedDistance >= distanceToLight) {
					//path is invalid for mvnee, set estimatorPDF to 0
					tl_estimatorPDFs[e] = 0.0;
					validMVNEEPath = false;
					break;
				}

				assert(seedDistance > 0.0f);
				vec3 seedVertex = forkVertex + seedDistance * omega;
				tl_seedVertices[s] = seedVertex;
				float distanceToPrevVertex = seedDistance - lastSeedDistance;
				assert(distanceToPrevVertex > 0.0f);
				tl_seedSegmentLengths[s] = distanceToPrevVertex;

				lastSeedDistance = seedDistance;
			}
			//last segment:
			tl_seedSegmentLengths[mvneeSegmentCount - 1] = distanceToLight - lastSeedDistance;

			/////////////////////////////////
			// Calculate Perturbation PDF
			/////////////////////////////////
			double perturbationPDF = 1.0;
			if (validMVNEEPath) {
				//uv-perturbation for all MVNEE-vertices but the light vertex!
				for (int s = 0; s < (mvneeSegmentCount - 1); s++) {
					vec3 perturbedVertex = path->getVertexPosition(e + 1 + s);
					vec3 forkToPerturbed = perturbedVertex - forkVertex;

					// calculate sigma for perturbation: first convolution formula:
					double leftSideFactor = 0.0;
					for (int x = 0; x <= s; x++) {
						leftSideFactor += ((double)tl_seedSegmentLengths[x] * (double)tl_seedSegmentLengths[x]);
					}
					assert(leftSideFactor > 0.0);
					leftSideFactor = sqrt(leftSideFactor);

					double rightSideFactor = 0.0;
					for (int x = s + 1; x < mvneeSegmentCount; x++) {
						rightSideFactor += ((double)tl_seedSegmentLengths[x] * (double)tl_seedSegmentLengths[x]);
					}
					assert(rightSideFactor > 0.0);
					rightSideFactor = sqrt(rightSideFactor);

					double sigma = calcGaussProduct(leftSideFactor * sigmaForHG, rightSideFactor * sigmaForHG);
					double finalGGXAlpha = sigma * 1.637618734; //multiply with Constants for conversion from gauss to ggx

					//update pdfs:
					vec3 seedToPerturbed = perturbedVertex - tl_seedVertices[s];
					float uDist = dot(seedToPerturbed, u);
					float vDist = dot(seedToPerturbed, v);
					float r2 = (uDist*uDist) + (vDist*vDist);
					double uvPerturbPDF = GGX_2D_PDF(r2, finalGGXAlpha); //pdf for u-v-plane perturbation

					float distanceToPrevVertex = tl_seedSegmentLengths[s];
					double segmentCountPDF = medium.mu_t * exp(-medium.mu_t * (double)distanceToPrevVertex); //pdf for seed distance sampling
					double combinedPDF = (segmentCountPDF * uvPerturbPDF);

					perturbationPDF *= combinedPDF;
				}

				//set final estimator pdf if everything is valid
				if (validMVNEEPath) {
					float lastSeedSegmentLength = tl_seedSegmentLengths[mvneeSegmentCount - 1];
					if (lastSeedSegmentLength >= 0.0f) {
						//contribute sampling for last segment: pdf for sampling distance >= distance to Light!
						perturbationPDF *= exp(-medium.mu_t * (double)lastSeedSegmentLength);

						bool validLightDistances = scene->calcDistancesToLights(tl_lightDistances, forkVertex);
						assert(validLightDistances);
						double lightVertexSamplingPDF = scene->getLightVertexSamplingPDF(tl_lightDistances, lightSourceIndex);
						double finalEstimatorPDF = pathTracingStartPDF * lightVertexSamplingPDF * perturbationPDF;
						if (isfinite(finalEstimatorPDF)) {
							tl_estimatorPDFs[e] = finalEstimatorPDF;
						}
						else {
							cout << "finalEstimatorPDF infinite!" << endl;
							tl_estimatorPDFs[e] = 0.0;
						}
					}
					else {
						tl_estimatorPDFs[e] = 0.0;
					}
				}
				else {
					tl_estimatorPDFs[e] = 0.0;
				}
			}
			else {
				tl_estimatorPDFs[e] = 0.0;
			}
		}
		else {
			tl_estimatorPDFs[e] = 0.0;
		}
	}

	///////////////////////////////
	// calculate MIS weight
	///////////////////////////////
	double finalPDF = tl_estimatorPDFs[estimatorIndex]; //pdf of the estimator that created the path
	if (finalPDF > 0.0) {
		float finalContribution = (float)(measurementContrib / finalPDF);
		if (isfinite(finalContribution)) {
			assert(finalContribution >= 0.0);

			const int estimatorCount = segmentCount;
			float misWeight = misPowerWeight(finalPDF, tl_estimatorPDFs, estimatorCount);
			vec3 finalPixel = misWeight * finalContribution * colorThroughput * hitLightSource->getEmissionIntensity(lightVertex, lastDirection);
			return finalPixel;
		}
	}

	return errorValue;
}


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
vec3 VolumeRenderer::pathTracing_MVNEE(const vec3& rayOrigin, const vec3& rayDir)
{
	//get thread local pre-reservers data for this thread:
	int threadID = omp_get_thread_num();
	Path* pathTracingPath = pathTracingPaths[threadID];
	vec3* tl_seedVertices = seedVertices[threadID];
	vec3* tl_perturbedVertices = perturbedVertices[threadID];
	float* tl_seedSegmentLengths = seedSegmentLengths[threadID];
	float* tl_lightDistances = distancesToLightSources[threadID];

	//add first Vertex
	PathVertex origin(rayOrigin, TYPE_ORIGIN, -1, rayDir, VERT_HOM_MEDIUM);
	pathTracingPath->addVertex(origin);

	vec3 finalPixel(0.0f);

	BSDF* lastBSDF = 0;

	vec3 currPosition = rayOrigin;
	vec3 currDir = rayDir;

	while (pathTracingPath->getSegmentLength() < rendering.MAX_SEGMENT_COUNT) {
		//sanity check assertions:
		assert(pathTracingPath->getVertexCount() > 0);
		PathVertex lastPathVertex;
		pathTracingPath->getVertex(pathTracingPath->getSegmentLength(), lastPathVertex);
		assert(lastPathVertex.vertexType != TYPE_MVNEE);
		pathTracingPath->getVertex(pathTracingPath->getVertexCount() - 1, lastPathVertex);
		assert(lastPathVertex.vertexType != TYPE_MVNEE);

		//sample free path Lenth
		double freePathLength = sampleFreePathLength(sample1DOpenInterval(threadID), medium.mu_t);

		//this is the starting vertex for MVNEE, depending on the intersection, it can either be a surface vertex or medium vertex
		PathVertex forkVertex;

		//intersect scene
		RTCRay ray;
		vec3 intersectionNormal;
		if (scene->intersectScene(currPosition, currDir, ray, intersectionNormal) && ray.tfar <= freePathLength) {
			if (ray.tfar > Constants::epsilon) {

				//////////////////////////////
				// Surface interaction
				//////////////////////////////
				vec3 surfaceVertex = currPosition + ray.tfar * currDir;
				forkVertex.vertex = surfaceVertex;
				forkVertex.vertexType = TYPE_SURFACE;
				forkVertex.geometryID = ray.geomID;
				forkVertex.surfaceNormal = intersectionNormal;
				pathTracingPath->addVertex(forkVertex);

				int hitLightIndex;
				if (scene->lightIntersected(ray.geomID, &hitLightIndex)) {
					//////////////////////////////
					// Light hit
					//////////////////////////////
					LightSource* hitLight = scene->lightSources[hitLightIndex];

					//normal culling
					if (hitLight->validHitDirection(currDir)) {
						//estimator Index for Path tracing is 0!
						vec3 finalContribution;
						if (pathTracingPath->getSegmentLength() > 1) {
							finalContribution = calcFinalWeightedContribution(pathTracingPath, 0, hitLightIndex);
						}
						else {
							finalContribution = hitLight->getEmissionIntensity(surfaceVertex, currDir);
						}

						finalPixel += finalContribution;
					}
					break;
				}

				//surface normal culling:
				if (dot(intersectionNormal, -currDir) <= 0.0f) {
					break;
				}

				BSDF* bsdfData = scene->getBSDF(ray.geomID);
				lastBSDF = bsdfData;

				//sample direction based on BRDF:
				vec3 newDir = bsdfData->sampleBSDFDirection(intersectionNormal, currDir, sample1D(threadID), sample1D(threadID)); 
				if (!bsdfData->validOutputDirection(intersectionNormal, newDir)) {
					break;
				}

				//update variables:
				currPosition = surfaceVertex + Constants::epsilon * intersectionNormal;
				currDir = newDir;
			}
			else {
				break;
			}
		}
		else {
			//////////////////////////////
			// Medium interaction
			//////////////////////////////
			vec3 nextScatteringVertex = currPosition + (float)freePathLength * currDir;
			pathTracingPath->addHomVolVertex(nextScatteringVertex, TYPE_MEDIUM_SCATTERING);

			//sample direction based on Henyey-Greenstein:
			vec3 newDir = sampleHenyeyGreensteinDirection(currDir, sample1D(threadID), sample1D(threadID), medium.hg_g_F);

			//update variables:
			forkVertex.vertex = nextScatteringVertex;
			forkVertex.vertexType = TYPE_MEDIUM_SCATTERING;
			forkVertex.geometryID = -1;
			forkVertex.surfaceNormal = vec3(0.0f);

			currPosition = nextScatteringVertex;
			currDir = newDir;
		}

		int lastPathTracingVertexIndex = pathTracingPath->getSegmentLength();
		if (lastPathTracingVertexIndex >= rendering.MAX_SEGMENT_COUNT) {
			break;
		}

		//////////////////////////////
		// MVNEE
		//////////////////////////////	
		//only perform MVNEE if last BRDF was not specular 
		if (forkVertex.vertexType != TYPE_SURFACE || (!lastBSDF->isDelta())) {

			//first sample a light vertex on a chosen light source
			int sampledLightIndex;

			bool validLightDistances = scene->calcDistancesToLights(tl_lightDistances, forkVertex.vertex);
			assert(validLightDistances);

			vec3 lightVertex = scene->sampleLightPosition(tl_lightDistances, sample1D(threadID), sample1D(threadID), sample1D(threadID), &sampledLightIndex);
			if (sampledLightIndex > -1) {
				LightSource* sampledLightSource = scene->lightSources[sampledLightIndex];
				vec3 forkToLight = lightVertex - forkVertex.vertex;
				float distanceToLight = length(forkToLight);
				if (distanceToLight > Constants::epsilon) {

					//only perform MVNEE if the expected segment count is smaller than MAX_MVNEE_SEGMENTS
					float expectedSegments = distanceToLight / medium.meanFreePathF;
					if (isfinite(expectedSegments) && expectedSegments > 0.0f && expectedSegments <= rendering.MAX_MVNEE_SEGMENTS_F) {

						vec3 omega = forkToLight / distanceToLight; //direction of the line to the light source

						vec3 u, v;
						coordinateSystem(omega, u, v); //build tangent frame: (same for every seed vertex)

						//sample the segment lengths for the seed path:
						int mvneeSegmentCount;

						bool validPath = sampleSeedSegmentLengths(distanceToLight, tl_seedSegmentLengths, &mvneeSegmentCount, threadID);

						//special case: surface normal culling for only one segment:
						if (validPath && mvneeSegmentCount == 1) {
							if (forkVertex.vertexType == TYPE_SURFACE) {
								float cosThetaSurface = dot(omega, forkVertex.surfaceNormal);
								if (cosThetaSurface <= 0.0f) {
									validPath = false;
								}
							}
							//light normal culling!
							if (!sampledLightSource->validHitDirection(omega)) {
								validPath = false;
							}
						}

						if (validPath && mvneeSegmentCount > 0) {
							//consider max segment length:
							if (pathTracingPath->getSegmentLength() + mvneeSegmentCount <= rendering.MAX_SEGMENT_COUNT) {

								//determine seed vertices based on sampled segment lengths:
								vec3 prevVertex = forkVertex.vertex;
								for (int s = 0; s < (mvneeSegmentCount - 1); s++) {
									vec3 newSeedVertex = prevVertex + tl_seedSegmentLengths[s] * omega;
									tl_seedVertices[s] = newSeedVertex;
									prevVertex = newSeedVertex;
								}
								//add light vertex
								tl_seedVertices[mvneeSegmentCount - 1] = lightVertex;

								////////////////////////////
								//  perturb all vertices but the lightDiskVertex:
								////////////////////////////
								vec3 previousPerturbedVertex = forkVertex.vertex;

								vec3 seedVertex;
								vec3 perturbedVertex;
								bool validMVNEEPath = true;
								for (int p = 0; p < (mvneeSegmentCount - 1); p++) {
									seedVertex = tl_seedVertices[p];

									//sanity check: distances to start and end may not be <= 0
									float distanceToLineEnd = length(lightVertex - seedVertex);
									float distanceToLineStart = length(seedVertex - forkVertex.vertex);
									if (distanceToLineEnd <= 0.0f || distanceToLineStart <= 0.0f) {
										validMVNEEPath = false;
										break;
									}

									// calculate sigma for perturbation: first convolution formula:
									double leftSideFactor = 0.0;
									for (int x = 0; x <= p; x++) {
										leftSideFactor += ((double)tl_seedSegmentLengths[x] * (double)tl_seedSegmentLengths[x]);
									}
									leftSideFactor = sqrt(leftSideFactor);
									assert(leftSideFactor > 0.0f);
									double rightSideFactor = 0.0;
									for (int x = p + 1; x < mvneeSegmentCount; x++) {
										rightSideFactor += ((double)tl_seedSegmentLengths[x] * (double)tl_seedSegmentLengths[x]);
									}
									rightSideFactor = sqrt(rightSideFactor);
									assert(rightSideFactor > 0.0f);

									double sigma = calcGaussProduct(leftSideFactor * sigmaForHG, rightSideFactor * sigmaForHG);
									double finalGGXAlpha = sigma * 1.637618734; //conversion with Constants factor

									//perform perturbation using ggx radius and uniform angle in u,v plane 
									perturbVertexGGX2D(finalGGXAlpha, u, v, seedVertex, &perturbedVertex, threadID);

									float maxT = tl_seedSegmentLengths[p];

									//for first perturbed vertex, check if perturbation violates the normal culling condition on the surface!
									if (p == 0 && forkVertex.vertexType == TYPE_SURFACE) {
										vec3 forkToFirstPerturbed = perturbedVertex - forkVertex.vertex;
										float firstDistToPerturbed = length(forkToFirstPerturbed);
										//if (firstDistToPerturbed >= Constants::epsilon) {
										if (firstDistToPerturbed > 0.0f) {
											forkToFirstPerturbed /= firstDistToPerturbed;
											float cosThetaSurface = dot(forkToFirstPerturbed, forkVertex.surfaceNormal);
											if (cosThetaSurface <= 0.0f) {
												validMVNEEPath = false;
												break;
											}
										}
										else {
											validMVNEEPath = false;
											break;
										}

										previousPerturbedVertex = forkVertex.vertex + Constants::epsilon * forkVertex.surfaceNormal;
										maxT -= Constants::epsilon;
										if (maxT <= 0.0f) {
											validMVNEEPath = false;
											break;
										}
									}

									//visibility check:							
									vec3 visibilityDirection = normalize(perturbedVertex - previousPerturbedVertex);

									RTCRay shadowRay;
									if (scene->vertexOccluded(previousPerturbedVertex, visibilityDirection, maxT, shadowRay)) {
										validMVNEEPath = false;
										break;
									}

									//add vertex to path
									pathTracingPath->addHomVolVertex(perturbedVertex, TYPE_MVNEE);
									previousPerturbedVertex = perturbedVertex;
								}

								if (validMVNEEPath) {
									vec3 lastSegmentDir = lightVertex - previousPerturbedVertex;
									float lastSegmentLength = length(lastSegmentDir);
									if (lastSegmentLength > 0.0f) {
										lastSegmentDir /= lastSegmentLength;

										//visibility check: first potential normal culling:
										if (sampledLightSource->validHitDirection(lastSegmentDir)) {

											if (mvneeSegmentCount == 1 && forkVertex.vertexType == TYPE_SURFACE) {
												previousPerturbedVertex = forkVertex.vertex + Constants::epsilon * forkVertex.surfaceNormal;
												if (lastSegmentLength - Constants::epsilon > 0.0f) {
													lastSegmentLength -= Constants::epsilon;
												}
											}


											//occlusion check:
											RTCRay lastShadowRay;
											if (!scene->vertexOccluded(previousPerturbedVertex, lastSegmentDir, lastSegmentLength, lastShadowRay)) {
												PathVertex lightVertexStruct(lightVertex, TYPE_MVNEE, -1, sampledLightSource->normal, VERT_HOM_MEDIUM);
												pathTracingPath->addVertex(lightVertexStruct);

												//calculate measurement constribution, pdf and mis weight:
												//estimator Index for MVNEE is index of last path tracing vertex
												//make sure the last path tracing vertex index is set correctly in order to use the correct PDF
												vec3 finalContribution = calcFinalWeightedContribution(pathTracingPath, lastPathTracingVertexIndex, sampledLightIndex);
												finalPixel += finalContribution;
											}
										}
									}
									else {
										cout << "last MVNEE segment length <= 0: " << lastSegmentLength << endl;
									}
								}
							}
							//delete all MVNEE vertices from the path construct, so path racing stays correct
							pathTracingPath->cutMVNEEVertices();
						}

					}
				}
			}
		}
	}

	//Path sampling finished: now reset Path, so it can be filled again
	pathTracingPath->reset();

	return finalPixel;
}


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
* @return: the MIS weighted contribution of the path
*/
vec3 VolumeRenderer::calcFinalWeightedContribution_FINAL(Path* path, int estimatorIndex, int lightSourceIndex, const int firstPossibleMVNEEEstimatorIndex, const double& currentMeasurementContrib, const vec3& currentColorThroughput)
{
	const int segmentCount = path->getSegmentLength();
	assert(path->getVertexCount() >= 2);
	assert(segmentCount <= rendering.MAX_SEGMENT_COUNT);
	assert(estimatorIndex >= 0);
	assert(estimatorIndex < segmentCount);
	vec3 errorValue(0.0f); //returned in case of errors

	LightSource* hitLightSource = scene->lightSources[lightSourceIndex];

	vec3 lastDirection;
	vec3 lightVertex = path->getVertexPosition(segmentCount);

	//special case: light was immediately hit after 1 segment:
	if (path->getVertexCount() == 2) {
		vec3 firstVertex = path->getVertexPosition(0);
		vec3 dir = normalize(lightVertex - firstVertex);
		return hitLightSource->getEmissionIntensity(lightVertex, dir);
	}


	//get thread local pre-reservers data for this thread:
	int threadID = omp_get_thread_num();	

	vec3* tl_seedVertices = seedVertices[threadID];
	double* tl_cumulatedPathTracingPDFs = cumulatedPathTracingPDFs[threadID];
	float* tl_seedSegmentLengths = seedSegmentLengths[threadID];
	double* tl_estimatorPDFs = estimatorPDFs[threadID];
	double* tl_seedSegmentLengthSquares = seedSegmentLengthSquares[threadID];
	float* tl_lightDistances = distancesToLightSources[threadID];

	//get measurement contrib and pdf and colorThroughput for the path using path tracing up to the last path tracing vertex:
	double measurementContrib = currentMeasurementContrib;
	double pathTracingPDF = tl_cumulatedPathTracingPDFs[segmentCount];
	vec3 colorThroughput = currentColorThroughput;


	/////////////////////////////////////////////////////////////
	// calculate measurementContribution as well as path tracing pdfs
	/////////////////////////////////////////////////////////////

	//in case the path wasn't created with path tracing, measurement contrib and pdf have to be calculated for the end of the path
	if (estimatorIndex > 0) {
		pathTracingPDF = tl_cumulatedPathTracingPDFs[estimatorIndex];

		// first direction can potentially be sampled by BRDF!!
		PathVertex vertex1, vertex2;
		path->getVertex(estimatorIndex - 1, vertex1);
		path->getVertex(estimatorIndex, vertex2);

		vec3 previousDir = vertex2.vertex - vertex1.vertex;
		float previousDistance = length(previousDir);
		if (previousDistance <= 0.0f) {
			return errorValue;
		}
		previousDir /= previousDistance;

		//calculate path tracing PDF and measurement contrib for every remaining vertex:
		PathVertex currVert, prevVert;
		for (int i = estimatorIndex + 1; i < path->getVertexCount(); i++) {
			path->getVertex(i, currVert);
			path->getVertex(i - 1, prevVert);
			vec3 currentDir = currVert.vertex - prevVert.vertex;
			float currentDistance = length(currentDir);
			if (currentDistance <= 0.0f) {
				return errorValue;
			}
			currentDir = currentDir / currentDistance; //normalize direction

			double currTransmittance = exp(-medium.mu_t * (double)currentDistance);
			double currG = 1.0 / ((double)(currentDistance)* (double)(currentDistance));
			assert(isfinite(currG));

			//contribute phase function * mu_s / BRDF depending on vertex
			if (prevVert.vertexType == TYPE_SURFACE) {
				//BRDF direction
				BSDF* bsdfData = scene->getBSDF(prevVert.geometryID);

				if (!bsdfData->validOutputDirection(prevVert.surfaceNormal, currentDir) ) {
					cout << "wrong outgoing direction at surface!" << endl; 
					return errorValue;
				}

				if (!bsdfData->isDelta()) {
					colorThroughput *= bsdfData->evalBSDF(prevVert.surfaceNormal, currentDir, previousDir);
					pathTracingPDF *= bsdfData->getBSDFDirectionPDF(prevVert.surfaceNormal, currentDir, previousDir);
				}
			}
			else {
				//phase direction
				float cosThetaPhase = dot(previousDir, currentDir);
				double phase = (double)henyeyGreenstein(cosThetaPhase, medium.hg_g_F);

				measurementContrib *= medium.mu_s * phase;
				pathTracingPDF *= phase;
			}

			if (i == segmentCount) {
				//Light connection segment:
				if (hitLightSource->validHitDirection(currentDir)) {
					lastDirection = currentDir;

					//special treatment when light source was hit:
					float cosLightF = dot(-currentDir, hitLightSource->normal);
					assert(cosLightF >= 0.0f);
					currG *= (double)cosLightF;

					measurementContrib *= currTransmittance * currG;
					pathTracingPDF *= currTransmittance * currG;
				}
				else {
					cout << "path hits light from backside!" << endl;
					return errorValue;
				}
			}
			//contribute G-Terms, transmittance: take care of medium coefficients!
			else {
				if (currVert.vertexType == TYPE_SURFACE) {
					float cosTheta = dot(currVert.surfaceNormal, -currentDir);
					if (cosTheta <= 0.0f) {
						return errorValue;
					}
					assert(cosTheta > 0.0f);
					currG *= (double)cosTheta;

					measurementContrib *= currTransmittance * currG;
					pathTracingPDF *= currTransmittance * currG;
				}
				else {
					measurementContrib *= currTransmittance * currG;
					pathTracingPDF *= medium.mu_t * currTransmittance * currG;
				}
			}

			tl_cumulatedPathTracingPDFs[i] = pathTracingPDF;
			previousDir = currentDir;
		}
	}
	else {
		vec3 preLastVertex = path->getVertexPosition(segmentCount - 1);
		lastDirection = normalize(lightVertex - preLastVertex);
	}

	//set path tracing pdf at index 0
	tl_estimatorPDFs[0] = pathTracingPDF;

	//////////////////////////////////////
	//calculate estimator PDFs:
	//////////////////////////////////////
	double sigmaForHgSquare = sigmaForHG * sigmaForHG;

	for (int e = firstPossibleMVNEEEstimatorIndex; e < segmentCount; e++) {
		double pathTracingStartPDF = tl_cumulatedPathTracingPDFs[e]; //pdf for sampling the first vertices with path tracing
		vec3 forkVertex = path->getVertexPosition(e); //last path tracing vertex and anchor point for the mvnee path

		vec3 forkToLight = lightVertex - forkVertex;
		float distanceToLight = length(forkToLight);

		float expectedSegments = distanceToLight / medium.meanFreePathF;

		if (distanceToLight <= 0.0f) {
			tl_estimatorPDFs[e] = 0.0;
		}
		else if (isfinite(expectedSegments) && expectedSegments > 0.0f && expectedSegments <= rendering.MAX_MVNEE_SEGMENTS_F) {
			//MVNEE is only valid if expected segment count is smaller than MAX_MVNEE_SEGMENTS!			
			vec3 omega = forkToLight / distanceToLight; //normalize direction

			//segment count of mvnee path, which is also the mvnee vertex count
			int mvneeSegmentCount = segmentCount - e;

			//build tangent frame: (same for every seed vertex)
			vec3 u, v;
			coordinateSystem(omega, u, v);

			/////////////////////////////////
			// recover seed distances and seed vertices:
			/////////////////////////////////
			bool validMVNEEPath = true;
			float lastSeedDistance = 0.0f;
			for (int s = 0; s < (mvneeSegmentCount - 1); s++) {
				vec3 perturbedVertex = path->getVertexPosition(e + 1 + s);
				vec3 forkToPerturbed = perturbedVertex - forkVertex;

				//use orthogonal projection to recover the seed distance:
				float seedDistance = dot(forkToPerturbed, omega);

				//now some things have to be checked in order to make sure the path can be created with mvnee:
				//the new Distance from forkVertex has to be larger than the summed distance,
				//also the summedDistance may never be larger than distanceToLight!!
				if (seedDistance <= 0.0f || seedDistance <= lastSeedDistance || seedDistance >= distanceToLight) {
					//path is invalid for mvnee, set estimatorPDF to 0
					tl_estimatorPDFs[e] = 0.0;
					validMVNEEPath = false;
					break;
				}

				assert(seedDistance > 0.0f);
				vec3 seedVertex = forkVertex + seedDistance * omega;
				tl_seedVertices[s] = seedVertex;
				float distanceToPrevVertex = seedDistance - lastSeedDistance;
				assert(distanceToPrevVertex > 0.0f);
				tl_seedSegmentLengths[s] = distanceToPrevVertex;
				tl_seedSegmentLengthSquares[s] = distanceToPrevVertex * distanceToPrevVertex;

				lastSeedDistance = seedDistance;
			}
			//last segment:
			float lastSegmentLength = distanceToLight - lastSeedDistance;
			tl_seedSegmentLengths[mvneeSegmentCount - 1] = lastSegmentLength;
			tl_seedSegmentLengthSquares[mvneeSegmentCount - 1] = (double)lastSegmentLength * (double)lastSegmentLength;

			/////////////////////////////////
			// Calculate Perturbation PDF
			/////////////////////////////////
			double perturbationPDF = 1.0;

			double leftSideFactor = 0.0;
			double rightSideFactor = 0.0;
			for (int x = 0; x < mvneeSegmentCount; x++) {
				rightSideFactor += tl_seedSegmentLengthSquares[x];
			}

			if (validMVNEEPath) {
				//uv-perturbation for all MVNEE-vertices but the light vertex!
				for (int s = 0; s < (mvneeSegmentCount - 1); s++) {
					vec3 perturbedVertex = path->getVertexPosition(e + 1 + s);
					vec3 forkToPerturbed = perturbedVertex - forkVertex;

					double currSegLengthSqu = tl_seedSegmentLengthSquares[s];
					leftSideFactor += currSegLengthSqu;
					rightSideFactor -= currSegLengthSqu;
					if (rightSideFactor <= 0.0f) {
						validMVNEEPath = false;
						break;
					}

					double sigma = calcGaussProductSquaredSigmas(leftSideFactor * sigmaForHgSquare, rightSideFactor * sigmaForHgSquare);
					double finalGGXAlpha = sigma * Constants::GGX_CONVERSION_Constant; //multiply with Constants for conversion from gauss to ggx

					//update pdfs:
					vec3 seedToPerturbed = perturbedVertex - tl_seedVertices[s];
					float uDist = dot(seedToPerturbed, u);
					float vDist = dot(seedToPerturbed, v);
					float r2 = (uDist*uDist) + (vDist*vDist);
					double uvPerturbPDF = GGX_2D_PDF(r2, finalGGXAlpha); //pdf for u-v-plane perturbation

					float distanceToPrevVertex = tl_seedSegmentLengths[s];

					double combinedPDF = (medium.mu_t * uvPerturbPDF);
					perturbationPDF *= combinedPDF;
				}

				//set final estimator pdf if everything is valid
				if (validMVNEEPath) {
					float lastSeedSegmentLength = tl_seedSegmentLengths[mvneeSegmentCount - 1];
					if (lastSeedSegmentLength >= 0.0f) {
						perturbationPDF *= exp(-medium.mu_t * (double)distanceToLight);

						bool validLightDistances = scene->calcDistancesToLights(tl_lightDistances, forkVertex);
						assert(validLightDistances);
						double lightVertexSamplingPDF = scene->getLightVertexSamplingPDF(tl_lightDistances, lightSourceIndex);

						double finalEstimatorPDF = pathTracingStartPDF * lightVertexSamplingPDF * perturbationPDF;
						if (isfinite(finalEstimatorPDF)) {
							tl_estimatorPDFs[e] = finalEstimatorPDF;
						}
						else {
							cout << "finalEstimatorPDF infinite!" << endl;
							tl_estimatorPDFs[e] = 0.0;
						}
					}
					else {
						tl_estimatorPDFs[e] = 0.0;
					}
				}
				else {
					tl_estimatorPDFs[e] = 0.0;
				}
			}
			else {
				tl_estimatorPDFs[e] = 0.0;
			}
		}
		else {
			tl_estimatorPDFs[e] = 0.0;
		}
	}

	///////////////////////////////
	// calculate MIS weight
	///////////////////////////////
	double finalPDF = tl_estimatorPDFs[estimatorIndex]; //pdf of the estimator that created the path
	if (finalPDF > 0.0) {
		float finalContribution = (float)(measurementContrib / finalPDF);
		if (isfinite(finalContribution)) {
			assert(finalContribution >= 0.0);

			int mvneeEstimatorCount = segmentCount - firstPossibleMVNEEEstimatorIndex;
			float misWeight = misBalanceWeight(finalPDF, pathTracingPDF, &tl_estimatorPDFs[firstPossibleMVNEEEstimatorIndex], mvneeEstimatorCount);

			vec3 finalPixel = misWeight * finalContribution * colorThroughput * hitLightSource->getEmissionIntensity(lightVertex, lastDirection);
			return finalPixel;
		}
	}

	return errorValue;
}


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
vec3 VolumeRenderer::pathTracing_MVNEE_FINAL(const vec3& rayOrigin, const vec3& rayDir)
{
	//get thread local pre-reservers data for this thread:
	int threadID = omp_get_thread_num();
	Path* pathTracingPath = pathTracingPaths[threadID];
	vec3* tl_seedVertices = seedVertices[threadID];
	vec3* tl_perturbedVertices = perturbedVertices[threadID];
	float* tl_seedSegmentLengths = seedSegmentLengths[threadID];
	double* tl_seedSegmentLengthSquares = seedSegmentLengthSquares[threadID];
	float* tl_lightDistances = distancesToLightSources[threadID];

	double* tl_cumulativePathTracingPDFs = cumulatedPathTracingPDFs[threadID];
	tl_cumulativePathTracingPDFs[0] = 1.0;

	//add first Vertex
	PathVertex origin(rayOrigin, TYPE_ORIGIN, -1, rayDir, VERT_HOM_MEDIUM);
	pathTracingPath->addVertex(origin);

	vec3 finalPixel(0.0f);

	vec3 currPosition = rayOrigin;
	vec3 currDir = rayDir;

	BSDF* lastBSDF = 0;

	double pathTracingPDF = 1.0;
	double measurementContrib = 1.0;
	double newMeasurementContrib = 1.0;
	vec3 colorThroughput(1.0f);
	vec3 newColorThroughput(1.0f);

	//index of first vertex which could serve as a MVNEE estimator starting point
	int firstPossibleMVNEEEstimatorIndex = 1;


	while (pathTracingPath->getSegmentLength() < rendering.MAX_SEGMENT_COUNT && isNotZero(colorThroughput)) {
		//sanity check assertions:
		assert(pathTracingPath->getVertexCount() > 0);
		PathVertex lastPathVertex;
		pathTracingPath->getVertex(pathTracingPath->getSegmentLength(), lastPathVertex);
		assert(lastPathVertex.vertexType != TYPE_MVNEE);
		pathTracingPath->getVertex(pathTracingPath->getVertexCount() - 1, lastPathVertex);
		assert(lastPathVertex.vertexType != TYPE_MVNEE);

		//sample free path Lenth
		double freePathLength = sampleFreePathLength(sample1DOpenInterval(threadID), medium.mu_t);

		//this is the starting vertex for MVNEE, depending on the intersection, it can either be a surface vertex or medium vertex
		PathVertex forkVertex;

		//intersect scene
		RTCRay ray;
		vec3 intersectionNormal;
		if (scene->intersectScene(currPosition, currDir, ray, intersectionNormal) && ray.tfar <= freePathLength) {
			if (ray.tfar > Constants::epsilon) {

				//////////////////////////////
				// Surface interaction
				//////////////////////////////
				vec3 surfaceVertex = currPosition + ray.tfar * currDir;
				forkVertex.vertex = surfaceVertex;
				forkVertex.vertexType = TYPE_SURFACE;
				forkVertex.geometryID = ray.geomID;
				forkVertex.surfaceNormal = intersectionNormal;
				pathTracingPath->addVertex(forkVertex);

				int hitLightIndex;
				if (scene->lightIntersected(ray.geomID, &hitLightIndex)) {
					//////////////////////////////
					// Light hit
					//////////////////////////////
					LightSource* hitLight = scene->lightSources[hitLightIndex];

					//normal culling
					if (hitLight->validHitDirection(currDir)) {

						//update contribution
						double transmittance = exp(-medium.mu_t * ray.tfar);
						float cosThetaLight = dot(-currDir, hitLight->normal);
						double G = cosThetaLight / ((double)ray.tfar * (double)ray.tfar);
						measurementContrib *= transmittance * G;
						pathTracingPDF *= transmittance * G;
						tl_cumulativePathTracingPDFs[pathTracingPath->getSegmentLength()] = pathTracingPDF;

						//estimator Index for Path tracing is 0!
						vec3 finalContribution;
						if (pathTracingPath->getSegmentLength() > 1) {
							finalContribution = calcFinalWeightedContribution_FINAL(pathTracingPath, 0, hitLightIndex, firstPossibleMVNEEEstimatorIndex, measurementContrib, colorThroughput);
						}
						else {
							finalContribution = hitLight->getEmissionIntensity(surfaceVertex, currDir);
						}

						finalPixel += finalContribution;
					}
					break;
				}

				//surface normal culling:
				if (dot(intersectionNormal, -currDir) <= 0.0f) {
					break;
				}
								
				//update pdf and mc:
				double transmittance = exp(-medium.mu_t * ray.tfar);
				float cosThetaSurface = dot(-currDir, intersectionNormal);
				double G = cosThetaSurface / ((double)ray.tfar * (double)ray.tfar);
				measurementContrib *= transmittance * G;
				pathTracingPDF *= transmittance * G;
				tl_cumulativePathTracingPDFs[pathTracingPath->getSegmentLength()] = pathTracingPDF;
				

				BSDF* bsdfData = scene->getBSDF(ray.geomID);
				lastBSDF = bsdfData;

				//sample direction based on BRDF:
				vec3 newDir = bsdfData->sampleBSDFDirection(intersectionNormal, currDir, sample1D(threadID), sample1D(threadID)); 
				if (!bsdfData->validOutputDirection(intersectionNormal, newDir)) {
					break;
				}
				if (!bsdfData->isDelta()) {

					//update index for MVNEE start vertex:
					firstPossibleMVNEEEstimatorIndex = pathTracingPath->getSegmentLength();

					//update pdf, measurement contrib and colorThroughput for BRDF-interaction:
					double dirSamplingPDF = bsdfData->getBSDFDirectionPDF(intersectionNormal, newDir, currDir);
					newMeasurementContrib = measurementContrib;
					pathTracingPDF *= dirSamplingPDF;
					newColorThroughput = colorThroughput * bsdfData->evalBSDF(intersectionNormal, newDir, currDir);
				}
				else {
					//update index for MVNEE start vertex:
					firstPossibleMVNEEEstimatorIndex = pathTracingPath->getSegmentLength() + 1;

					newMeasurementContrib = measurementContrib;
					newColorThroughput = colorThroughput;
				}

				//update variables:
				currPosition = surfaceVertex + Constants::epsilon * intersectionNormal;
				currDir = newDir;
			}
			else {
				break;
			}
		}
		else {
			//////////////////////////////
			// Medium interaction
			//////////////////////////////
			vec3 nextScatteringVertex = currPosition + (float)freePathLength * currDir;
			pathTracingPath->addHomVolVertex(nextScatteringVertex, TYPE_MEDIUM_SCATTERING);


			//update pdf and mc:
			double transmittance = exp(-medium.mu_t * freePathLength);
			double G = 1.0 / ((double)freePathLength * (double)freePathLength);
			if (!isfinite(G)) {
				break;
			}
			assert(isfinite(G));
			measurementContrib *= transmittance * G;
			pathTracingPDF *= medium.mu_t * transmittance * G;
			tl_cumulativePathTracingPDFs[pathTracingPath->getSegmentLength()] = pathTracingPDF;


			//sample direction based on Henyey-Greenstein:
			vec3 newDir = sampleHenyeyGreensteinDirection(currDir, sample1D(threadID), sample1D(threadID), medium.hg_g_F);

			//update pdf and mc for phase function
			float cos_theta_Phase = dot(currDir, newDir);
			double phase = (double)henyeyGreenstein(cos_theta_Phase, medium.hg_g_F);
			newMeasurementContrib = measurementContrib * medium.mu_s * phase;
			pathTracingPDF *= phase;

			//update variables:
			forkVertex.vertex = nextScatteringVertex;
			forkVertex.vertexType = TYPE_MEDIUM_SCATTERING;
			forkVertex.geometryID = -1;
			forkVertex.surfaceNormal = vec3(0.0f);

			currPosition = nextScatteringVertex;
			currDir = newDir;
		}

		int lastPathTracingVertexIndex = pathTracingPath->getSegmentLength();
		if (lastPathTracingVertexIndex >= rendering.MAX_SEGMENT_COUNT) {
			break;
		}

		//////////////////////////////
		// MVNEE
		//////////////////////////////		

		//only perform MVNEE on surfaces other than delta BSDFs
		if (forkVertex.vertexType != TYPE_SURFACE || (lastBSDF != 0 && !lastBSDF->isDelta())) {

			//first sample a light vertex
			int sampledLightIndex;

			bool validLightDistances = scene->calcDistancesToLights(tl_lightDistances, forkVertex.vertex);
			assert(validLightDistances);
			vec3 lightVertex = scene->sampleLightPosition(tl_lightDistances, sample1D(threadID), sample1D(threadID), sample1D(threadID), &sampledLightIndex);
			if (sampledLightIndex > -1) {
				LightSource* sampledLightSource = scene->lightSources[sampledLightIndex];
				vec3 forkToLight = lightVertex - forkVertex.vertex;
				float distanceToLight = length(forkToLight);
				if (distanceToLight > Constants::epsilon) {

					//only perform MVNEE if the expected segment count is smaller than MAX_MVNEE_SEGMENTS
					float expectedSegments = distanceToLight / medium.meanFreePathF;
					if (isfinite(expectedSegments) && expectedSegments > 0.0f && expectedSegments <= rendering.MAX_MVNEE_SEGMENTS_F) {

						vec3 omega = forkToLight / distanceToLight; //direction of the line to the light source

						vec3 u, v;
						coordinateSystem(omega, u, v); //build tangent frame: (same for every seed vertex)

						//sample the segment lengths for the seed path:
						int mvneeSegmentCount;

						bool validPath = sampleSeedSegmentLengths(distanceToLight, tl_seedSegmentLengths, tl_seedSegmentLengthSquares, &mvneeSegmentCount, threadID);

						//special case: surface normal culling for only one segment:
						if (validPath && mvneeSegmentCount == 1) {
							if (forkVertex.vertexType == TYPE_SURFACE) {
								float cosThetaSurface = dot(omega, forkVertex.surfaceNormal);
								if (cosThetaSurface <= 0.0f) {
									validPath = false;
								}
							}
							//light normal culling!
							if (!sampledLightSource->validHitDirection(omega)) {
								validPath = false;
							}
						}

						if (validPath && mvneeSegmentCount > 0) {
							//consider max segment length:
							if (pathTracingPath->getSegmentLength() + mvneeSegmentCount <= rendering.MAX_SEGMENT_COUNT) {

								//determine seed vertices based on sampled segment lengths:
								vec3 prevVertex = forkVertex.vertex;
								for (int s = 0; s < (mvneeSegmentCount - 1); s++) {
									vec3 newSeedVertex = prevVertex + tl_seedSegmentLengths[s] * omega;
									tl_seedVertices[s] = newSeedVertex;
									prevVertex = newSeedVertex;
								}
								//add light vertex
								tl_seedVertices[mvneeSegmentCount - 1] = lightVertex;

								////////////////////////////
								//  perturb all vertices but the lightDiskVertex:
								////////////////////////////
								vec3 previousPerturbedVertex = forkVertex.vertex;
								double sigmaForHgSquare = sigmaForHG * sigmaForHG;

								vec3 seedVertex;
								vec3 perturbedVertex;
								bool validMVNEEPath = true;

								double leftSideFactor = 0.0;
								double rightSideFactor = 0.0;
								for (int x = 0; x < mvneeSegmentCount; x++) {
									rightSideFactor += tl_seedSegmentLengthSquares[x];
								}

								for (int p = 0; p < (mvneeSegmentCount - 1); p++) {
									seedVertex = tl_seedVertices[p];

									//sanity check: distances to start and end may not be <= 0
									float distanceToLineEnd = length(lightVertex - seedVertex);
									float distanceToLineStart = length(seedVertex - forkVertex.vertex);
									if (distanceToLineEnd <= 0.0f || distanceToLineStart <= 0.0f) {
										validMVNEEPath = false;
										break;
									}

									double currSegLengthSqr = tl_seedSegmentLengthSquares[p];
									leftSideFactor += currSegLengthSqr;
									rightSideFactor -= currSegLengthSqr;
									if (rightSideFactor <= 0.0f) {
										validMVNEEPath = false;
										break;
									}

									double sigma = calcGaussProductSquaredSigmas(leftSideFactor * sigmaForHgSquare, rightSideFactor * sigmaForHgSquare);
									double finalGGXAlpha = sigma * Constants::GGX_CONVERSION_Constant; //conversion with Constants factor

									//perform perturbation using ggx radius and uniform angle in u,v plane 
									perturbVertexGGX2D(finalGGXAlpha, u, v, seedVertex, &perturbedVertex, threadID);

									float maxT = tl_seedSegmentLengths[p];

									//for first perturbed vertex, check if perturbation violates the normal culling condition on the surface!
									if (p == 0 && forkVertex.vertexType == TYPE_SURFACE) {
										vec3 forkToFirstPerturbed = perturbedVertex - forkVertex.vertex;
										float firstDistToPerturbed = length(forkToFirstPerturbed);
										if (firstDistToPerturbed > 0.0f) {
											forkToFirstPerturbed /= firstDistToPerturbed;
											float cosThetaSurface = dot(forkToFirstPerturbed, forkVertex.surfaceNormal);
											if (cosThetaSurface <= 0.0f) {
												validMVNEEPath = false;
												break;
											}
										}
										else {
											validMVNEEPath = false;
											break;
										}

										previousPerturbedVertex = forkVertex.vertex + Constants::epsilon * forkVertex.surfaceNormal;

										maxT -= Constants::epsilon;
										if (maxT <= 0.0f) {
											validMVNEEPath = false;
											break;
										}
									}

									//visibility check:

									vec3 visibilityDirection = normalize(perturbedVertex - previousPerturbedVertex);

									RTCRay shadowRay;
									if (scene->vertexOccluded(previousPerturbedVertex, visibilityDirection, maxT, shadowRay)) {
										validMVNEEPath = false;
										break;
									}

									//add vertex to path
									pathTracingPath->addHomVolVertex(perturbedVertex, TYPE_MVNEE);
									previousPerturbedVertex = perturbedVertex;
								}

								if (validMVNEEPath) {
									vec3 lastSegmentDir = lightVertex - previousPerturbedVertex;
									float lastSegmentLength = length(lastSegmentDir);
									if (lastSegmentLength > 0.0f) {
										assert(lastSegmentLength > 0.0f);
										lastSegmentDir /= lastSegmentLength;

										//visibility check: first potential normal culling:
										if (sampledLightSource->validHitDirection(lastSegmentDir)) {

											if (mvneeSegmentCount == 1 && forkVertex.vertexType == TYPE_SURFACE) {
												previousPerturbedVertex = forkVertex.vertex + Constants::epsilon * forkVertex.surfaceNormal;

												if (lastSegmentLength - Constants::epsilon > 0.0f) {
													lastSegmentLength -= Constants::epsilon;
												}
											}

											//occlusion check:
											RTCRay lastShadowRay;
											if (!scene->vertexOccluded(previousPerturbedVertex, lastSegmentDir, lastSegmentLength, lastShadowRay)) {
												PathVertex lightVertexStruct(lightVertex, TYPE_MVNEE, -1, sampledLightSource->normal, VERT_HOM_MEDIUM);
												pathTracingPath->addVertex(lightVertexStruct);

												//calculate measurement constribution, pdf and mis weight:
												//estimator Index for MVNEE is index of last path tracing vertex
												//make sure the last path tracing vertex index is set correctly in order to use the correct PDF
												vec3 finalContribution = calcFinalWeightedContribution_FINAL(pathTracingPath, lastPathTracingVertexIndex, sampledLightIndex, firstPossibleMVNEEEstimatorIndex, measurementContrib, colorThroughput);
												finalPixel += finalContribution;
											}
										}
									}
									else {
										//cout << "last MVNEE segment length <= 0: " << lastSegmentLength << endl;
									}
								}
							}
							//delete all MVNEE vertices from the path construct, so path racing stays correct
							pathTracingPath->cutMVNEEVertices();
						}

					}
				}
			}
		}
		measurementContrib = newMeasurementContrib;
		colorThroughput = newColorThroughput;
	}

	//Path sampling finished: now reset Path, so it can be filled again
	pathTracingPath->reset();

	return finalPixel;
}


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
* @return: the MIS weighted contribution of the path
*/
vec3 VolumeRenderer::calcFinalWeightedContribution_LightImportanceSampling(Path* path, int estimatorIndex, int lightSourceIndex, const int firstPossibleMVNEEEstimatorIndex, const double& currentMeasurementContrib, const vec3& currentColorThroughput)
{
	const int segmentCount = path->getSegmentLength();
	assert(path->getVertexCount() >= 2);
	assert(segmentCount <= rendering.MAX_SEGMENT_COUNT);
	assert(estimatorIndex >= 0);
	assert(estimatorIndex < segmentCount);
	vec3 errorValue(0.0f); //returned in case of errors

	LightSource* hitLightSource = scene->lightSources[lightSourceIndex];

	float segmentLengthAtLight = 0.0f;
	vec3 lastDirection;
	vec3 lightVertex = path->getVertexPosition(segmentCount);

	//special case: light was immediately hit after 1 segment:
	if (path->getVertexCount() == 2) {
		vec3 firstVertex = path->getVertexPosition(0);
		vec3 dir = normalize(lightVertex - firstVertex);
		return hitLightSource->getEmissionIntensity(lightVertex, dir);
	}


	//get thread local pre-reserved data for this thread:
	int threadID = omp_get_thread_num();

	vec3* tl_seedVertices = seedVertices[threadID];
	double* tl_cumulatedPathTracingPDFs = cumulatedPathTracingPDFs[threadID];
	float* tl_seedSegmentLengths = seedSegmentLengths[threadID];
	double* tl_estimatorPDFs = estimatorPDFs[threadID];
	double* tl_seedSegmentLengthSquares = seedSegmentLengthSquares[threadID];
	float* tl_lightDistances = distancesToLightSources[threadID];

	//get measurement contrib and pdf and colorThroughput for the path using path tracing up to the last path tracing vertex:
	double measurementContrib = currentMeasurementContrib;
	double pathTracingPDF = tl_cumulatedPathTracingPDFs[segmentCount];
	vec3 colorThroughput = currentColorThroughput;


	/////////////////////////////////////////////////////////////
	// calculate measurementContribution as well as path tracing pdfs
	/////////////////////////////////////////////////////////////

	//in case the path wasn't created with path tracing, measurement contrib and pdf have to be calculated for the end of the path
	if (estimatorIndex > 0) {
		pathTracingPDF = tl_cumulatedPathTracingPDFs[estimatorIndex];

		// first direction can potentially be sampled by BRDF!!
		PathVertex vertex1, vertex2;
		path->getVertex(estimatorIndex - 1, vertex1);
		path->getVertex(estimatorIndex, vertex2);

		vec3 previousDir = vertex2.vertex - vertex1.vertex;
		float previousDistance = length(previousDir);
		if (previousDistance <= 0.0f) {
			return errorValue;
		}
		previousDir /= previousDistance;

		//calculate path tracing PDF and measurement contrib for every remaining vertex:
		PathVertex currVert, prevVert;
		for (int i = estimatorIndex + 1; i < path->getVertexCount(); i++) {
			path->getVertex(i, currVert);
			path->getVertex(i - 1, prevVert);
			vec3 currentDir = currVert.vertex - prevVert.vertex;
			float currentDistance = length(currentDir);
			if (currentDistance <= 0.0f) {
				return errorValue;
			}
			currentDir = currentDir / currentDistance; //normalize direction

			double currTransmittance = exp(-medium.mu_t * (double)currentDistance);
			double currG = 1.0 / ((double)(currentDistance)* (double)(currentDistance));
			assert(isfinite(currG));

			//contribute phase function * mu_s / BRDF depending on vertex
			if (prevVert.vertexType == TYPE_SURFACE) {
				//BRDF direction
				BSDF* bsdfData = scene->getBSDF(prevVert.geometryID);

				//float cosThetaBRDF = dot(prevVert.surfaceNormal, currentDir);
				if (!bsdfData->validOutputDirection(prevVert.surfaceNormal, currentDir)) {
					cout << "wrong outgoing direction at surface!" << endl; 
					return errorValue;
				}

				if (!bsdfData->isDelta()) {
					colorThroughput *= bsdfData->evalBSDF(prevVert.surfaceNormal, currentDir, previousDir);
					pathTracingPDF *= bsdfData->getBSDFDirectionPDF(prevVert.surfaceNormal, currentDir, previousDir);
				}
			}
			else {
				//phase direction
				float cosThetaPhase = dot(previousDir, currentDir);
				double phase = (double)henyeyGreenstein(cosThetaPhase, medium.hg_g_F);

				measurementContrib *= medium.mu_s * phase;
				pathTracingPDF *= phase;
			}

			if (i == segmentCount) {
				//Light connection segment:
				if (hitLightSource->validHitDirection(currentDir)) {
					segmentLengthAtLight = currentDistance;
					lastDirection = currentDir;

					//special treatment when light source was hit:
					float cosLightF = dot(-currentDir, hitLightSource->normal);
					assert(cosLightF >= 0.0f);
					currG *= (double)cosLightF;

					measurementContrib *= currTransmittance * currG;
					pathTracingPDF *= currTransmittance * currG;
				}
				else {
					cout << "path hits light from backside!" << endl;
					return errorValue;
				}
			}
			//contribute G-Terms, transmittance: take care of medium coefficients!
			else {
				if (currVert.vertexType == TYPE_SURFACE) {
					float cosTheta = dot(currVert.surfaceNormal, -currentDir);
					if (cosTheta <= 0.0f) {
						return errorValue;
					}
					assert(cosTheta > 0.0f);
					currG *= (double)cosTheta;

					measurementContrib *= currTransmittance * currG;
					pathTracingPDF *= currTransmittance * currG;
				}
				else {
					measurementContrib *= currTransmittance * currG;
					pathTracingPDF *= medium.mu_t * currTransmittance * currG;
				}
			}

			tl_cumulatedPathTracingPDFs[i] = pathTracingPDF;
			previousDir = currentDir;
		}
	}
	else {
		vec3 preLastVertex = path->getVertexPosition(segmentCount - 1);
		lastDirection = lightVertex - preLastVertex;
		segmentLengthAtLight = length(lastDirection);
		if (segmentLengthAtLight <= 0.0f) {
			return errorValue;
		}
		lastDirection /= segmentLengthAtLight;
	}

	//set path tracing pdf at index 0
	tl_estimatorPDFs[0] = pathTracingPDF;

	//special cases: 
	// 1) path has three vertices: this path can only be created with path Tracing!
	// 2) Light was hit directly after a surface interaction -> only Path tracing can create this path
	if (segmentCount == 2 || firstPossibleMVNEEEstimatorIndex == segmentCount - 1) {
		float finalContribution = (float)(measurementContrib / pathTracingPDF);
		if (isfinite(finalContribution)) {
			assert(finalContribution >= 0.0);

			vec3 emissionIntensity = hitLightSource->getEmissionIntensity(lightVertex, lastDirection);
			vec3 finalPixel = finalContribution * colorThroughput * emissionIntensity;
			return finalPixel;

			//TEST: makes the problem visible: these paths, that cannot be created with MVNEE lead to severe noise!
			//return errorValue;
		}
		else {
			return errorValue;
		}
	}

	//////////////////////////////////////
	//calculate estimator PDFs:
	//////////////////////////////////////
	double sigmaForHgSquare = sigmaForHG * sigmaForHG;

	vec3 mvneeDestination = path->getVertexPosition(segmentCount - 1);

	//precalculate all the PDFs for the sampled light segment
	vec3 sampledLightDir = -lastDirection;

	//again trace ray to find the maximum sampling distance
	RTCRay lightRay;
	vec3 hitSurfaceNormal;
	scene->intersectScene(lightVertex + Constants::epsilon * hitLightSource->normal, sampledLightDir, lightRay, hitSurfaceNormal);

	double lightDirectionPDF = hitLightSource->getDirectionSamplingPDF(sampledLightDir) / ((double)segmentLengthAtLight * (double)segmentLengthAtLight);
	double lightDistancePDF = getLimitedFreePathPDF(segmentLengthAtLight, medium.mu_t, Constants::epsilon, lightRay.tfar - Constants::epsilon);

	//there is one less MVNEE estimator, since a one segment connection is no longer possible!
	for (int e = firstPossibleMVNEEEstimatorIndex; e < (segmentCount-1); e++) {
		double pathTracingStartPDF = tl_cumulatedPathTracingPDFs[e]; //pdf for sampling the first vertices with path tracing
		vec3 forkVertex = path->getVertexPosition(e); //last path tracing vertex and anchor point for the mvnee path

		vec3 forkToDestination = mvneeDestination - forkVertex;
		float distanceToDestination = length(forkToDestination);

		float expectedSegments = distanceToDestination / medium.meanFreePathF;

		if (distanceToDestination <= 0.0f) {
			tl_estimatorPDFs[e] = 0.0;
		}
		else if (isfinite(expectedSegments) && expectedSegments > 0.0f && expectedSegments <= rendering.MAX_MVNEE_SEGMENTS_F) {
			//MVNEE is only valid if expected segment count is smaller than MAX_MVNEE_SEGMENTS!			
			vec3 omega = forkToDestination / distanceToDestination; //normalize direction

			//segment count of mvnee path, which is also the mvnee vertex count (from fork vertex to mvneeDestination!)
			int mvneeSegmentCount = segmentCount - e - 1;

			//build tangent frame: (same for every seed vertex)
			vec3 u, v;
			coordinateSystem(omega, u, v);

			/////////////////////////////////
			// recover seed distances and seed vertices:
			/////////////////////////////////
			bool validMVNEEPath = true;
			float lastSeedDistance = 0.0f;
			for (int s = 0; s < (mvneeSegmentCount - 1); s++) {
				vec3 perturbedVertex = path->getVertexPosition(e + 1 + s);
				vec3 forkToPerturbed = perturbedVertex - forkVertex;

				//use orthogonal projection to recover the seed distance:
				float seedDistance = dot(forkToPerturbed, omega);

				//now some things have to be checked in order to make sure the path can be created with mvnee:
				//the new Distance from forkVertex has to be larger than the summed distance,
				//also the summedDistance may never be larger than distanceToLight!!
				if (seedDistance <= 0.0f || seedDistance <= lastSeedDistance || seedDistance >= distanceToDestination) {
					//path is invalid for mvnee, set estimatorPDF to 0
					tl_estimatorPDFs[e] = 0.0;
					validMVNEEPath = false;
					break;
				}

				assert(seedDistance > 0.0f);
				vec3 seedVertex = forkVertex + seedDistance * omega;
				tl_seedVertices[s] = seedVertex;
				float distanceToPrevVertex = seedDistance - lastSeedDistance;
				assert(distanceToPrevVertex > 0.0f);
				tl_seedSegmentLengths[s] = distanceToPrevVertex;
				tl_seedSegmentLengthSquares[s] = distanceToPrevVertex * distanceToPrevVertex;

				lastSeedDistance = seedDistance;
			}
			//last segment:
			float lastSegmentLength = distanceToDestination - lastSeedDistance;
			tl_seedSegmentLengths[mvneeSegmentCount - 1] = lastSegmentLength;
			tl_seedSegmentLengthSquares[mvneeSegmentCount - 1] = (double)lastSegmentLength * (double)lastSegmentLength;

			/////////////////////////////////
			// Calculate Perturbation PDF
			/////////////////////////////////
			double perturbationPDF = 1.0;

			double leftSideFactor = 0.0;
			double rightSideFactor = 0.0;
			for (int x = 0; x < mvneeSegmentCount; x++) {
				rightSideFactor += tl_seedSegmentLengthSquares[x];
			}

			if (validMVNEEPath) {
				//uv-perturbation for all MVNEE-vertices but the light vertex!
				for (int s = 0; s < (mvneeSegmentCount - 1); s++) {
					vec3 perturbedVertex = path->getVertexPosition(e + 1 + s);
					vec3 forkToPerturbed = perturbedVertex - forkVertex;

					double currSegLengthSqu = tl_seedSegmentLengthSquares[s];
					leftSideFactor += currSegLengthSqu;
					rightSideFactor -= currSegLengthSqu;
					if (rightSideFactor <= 0.0f) {
						validMVNEEPath = false;
						break;
					}

					double sigma = calcGaussProductSquaredSigmas(leftSideFactor * sigmaForHgSquare, rightSideFactor * sigmaForHgSquare);
					double finalGGXAlpha = sigma * Constants::GGX_CONVERSION_Constant; //multiply with Constants for conversion from gauss to ggx

					//update pdfs:
					vec3 seedToPerturbed = perturbedVertex - tl_seedVertices[s];
					float uDist = dot(seedToPerturbed, u);
					float vDist = dot(seedToPerturbed, v);
					float r2 = (uDist*uDist) + (vDist*vDist);
					double uvPerturbPDF = GGX_2D_PDF(r2, finalGGXAlpha); //pdf for u-v-plane perturbation

					float distanceToPrevVertex = tl_seedSegmentLengths[s];

					double combinedPDF = (medium.mu_t * uvPerturbPDF);
					perturbationPDF *= combinedPDF;
				}

				//set final estimator pdf if everything is valid
				if (validMVNEEPath) {
					float lastSeedSegmentLength = tl_seedSegmentLengths[mvneeSegmentCount - 1];
					if (lastSeedSegmentLength >= 0.0f) {
						perturbationPDF *= exp(-medium.mu_t * (double)distanceToDestination);

						bool validLightDistances = scene->calcDistancesToLights(tl_lightDistances, forkVertex);
						assert(validLightDistances);
						double lightVertexSamplingPDF = scene->getLightVertexSamplingPDF(tl_lightDistances, lightSourceIndex);
						double combinedLightPDF = lightVertexSamplingPDF * lightDirectionPDF * lightDistancePDF;

						double finalEstimatorPDF = pathTracingStartPDF * combinedLightPDF * perturbationPDF;
						if (isfinite(finalEstimatorPDF)) {
							tl_estimatorPDFs[e] = finalEstimatorPDF;
						}
						else {
							cout << "finalEstimatorPDF infinite!" << endl;
							tl_estimatorPDFs[e] = 0.0;
						}
					}
					else {
						tl_estimatorPDFs[e] = 0.0;
					}
				}
				else {
					tl_estimatorPDFs[e] = 0.0;
				}
			}
			else {
				tl_estimatorPDFs[e] = 0.0;
			}
		}
		else {
			tl_estimatorPDFs[e] = 0.0;
		}
	}

	///////////////////////////////
	// calculate MIS weight
	///////////////////////////////
	double finalPDF = tl_estimatorPDFs[estimatorIndex]; //pdf of the estimator that created the path
	if (finalPDF > 0.0) {
		float finalContribution = (float)(measurementContrib / finalPDF);
		if (isfinite(finalContribution)) {
			assert(finalContribution >= 0.0);

			//one less mvnee estimator, since the minimum segment count for MVNEE is now 2!
			int mvneeEstimatorCount = segmentCount - firstPossibleMVNEEEstimatorIndex - 1;
			float misWeight = misBalanceWeight(finalPDF, pathTracingPDF, &tl_estimatorPDFs[firstPossibleMVNEEEstimatorIndex], mvneeEstimatorCount);

			vec3 finalPixel = misWeight * finalContribution * colorThroughput * hitLightSource->getEmissionIntensity(lightVertex, lastDirection);
			return finalPixel;
		}
	}

	return errorValue;
}


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
vec3 VolumeRenderer::pathTracing_MVNEE_LightImportanceSampling(const vec3& rayOrigin, const vec3& rayDir)
{
	//get thread local pre-reservers data for this thread:
	int threadID = omp_get_thread_num();
	Path* pathTracingPath = pathTracingPaths[threadID];
	vec3* tl_seedVertices = seedVertices[threadID];
	vec3* tl_perturbedVertices = perturbedVertices[threadID];
	float* tl_seedSegmentLengths = seedSegmentLengths[threadID];
	double* tl_seedSegmentLengthSquares = seedSegmentLengthSquares[threadID];
	float* tl_lightDistances = distancesToLightSources[threadID];

	double* tl_cumulativePathTracingPDFs = cumulatedPathTracingPDFs[threadID];
	tl_cumulativePathTracingPDFs[0] = 1.0;

	//add first Vertex
	PathVertex origin(rayOrigin, TYPE_ORIGIN, -1, rayDir, VERT_HOM_MEDIUM);
	pathTracingPath->addVertex(origin);

	vec3 finalPixel(0.0f);
	BSDF* lastBSDF = 0;

	vec3 currPosition = rayOrigin;
	vec3 currDir = rayDir;

	double pathTracingPDF = 1.0;
	double measurementContrib = 1.0;
	double newMeasurementContrib = 1.0;
	vec3 colorThroughput(1.0f);
	vec3 newColorThroughput(1.0f);

	//index of first vertex which could serve as a MVNEE estimator starting point
	int firstPossibleMVNEEEstimatorIndex = 1;


	while (pathTracingPath->getSegmentLength() < rendering.MAX_SEGMENT_COUNT && isNotZero(colorThroughput)) {
		//sanity check assertions:
		assert(pathTracingPath->getVertexCount() > 0);
		PathVertex lastPathVertex;
		pathTracingPath->getVertex(pathTracingPath->getSegmentLength(), lastPathVertex);
		assert(lastPathVertex.vertexType != TYPE_MVNEE);
		pathTracingPath->getVertex(pathTracingPath->getVertexCount() - 1, lastPathVertex);
		assert(lastPathVertex.vertexType != TYPE_MVNEE);

		//sample free path Lenth
		double freePathLength = sampleFreePathLength(sample1DOpenInterval(threadID), medium.mu_t);

		//this is the starting vertex for MVNEE, depending on the intersection, it can either be a surface vertex or medium vertex
		PathVertex forkVertex;

		//intersect scene
		RTCRay ray;
		vec3 intersectionNormal;
		if (scene->intersectScene(currPosition, currDir, ray, intersectionNormal) && ray.tfar <= freePathLength) {
			if (ray.tfar > Constants::epsilon) {

				//////////////////////////////
				// Surface interaction
				//////////////////////////////
				vec3 surfaceVertex = currPosition + ray.tfar * currDir;
				forkVertex.vertex = surfaceVertex;
				forkVertex.vertexType = TYPE_SURFACE;
				forkVertex.geometryID = ray.geomID;
				forkVertex.surfaceNormal = intersectionNormal;
				pathTracingPath->addVertex(forkVertex);

				int hitLightIndex;
				if (scene->lightIntersected(ray.geomID, &hitLightIndex)) {
					//////////////////////////////
					// Light hit
					//////////////////////////////
					LightSource* hitLight = scene->lightSources[hitLightIndex];

					//normal culling
					if (hitLight->validHitDirection(currDir)) {

						//update contribution
						double transmittance = exp(-medium.mu_t * ray.tfar);
						float cosThetaLight = dot(-currDir, hitLight->normal);
						double G = cosThetaLight / ((double)ray.tfar * (double)ray.tfar);
						measurementContrib *= transmittance * G;
						pathTracingPDF *= transmittance * G;
						tl_cumulativePathTracingPDFs[pathTracingPath->getSegmentLength()] = pathTracingPDF;

						//estimator Index for Path tracing is 0!
						vec3 finalContribution;
						if (pathTracingPath->getSegmentLength() > 1) {
							finalContribution = calcFinalWeightedContribution_LightImportanceSampling(pathTracingPath, 0, hitLightIndex, firstPossibleMVNEEEstimatorIndex, measurementContrib, colorThroughput);
						}
						else {
							finalContribution = hitLight->getEmissionIntensity(surfaceVertex, currDir);
						}

						finalPixel += finalContribution;
					}
					break;
				}

				//surface normal culling:
				if (dot(intersectionNormal, -currDir) <= 0.0f) {
					break;
				}


				//update pdf and mc:
				double transmittance = exp(-medium.mu_t * ray.tfar);
				float cosThetaSurface = dot(-currDir, intersectionNormal);
				double G = cosThetaSurface / ((double)ray.tfar * (double)ray.tfar);
				measurementContrib *= transmittance * G;
				pathTracingPDF *= transmittance * G;
				tl_cumulativePathTracingPDFs[pathTracingPath->getSegmentLength()] = pathTracingPDF;


				BSDF* bsdfData = scene->getBSDF(ray.geomID);
				lastBSDF = bsdfData;

				//sample direction based on BRDF:
				vec3 newDir = bsdfData->sampleBSDFDirection(intersectionNormal, currDir, sample1D(threadID), sample1D(threadID));
				if (!bsdfData->validOutputDirection(intersectionNormal, newDir)) {
					break;
				}
				if (!bsdfData->isDelta()) {

					//update index for MVNEE start vertex:
					firstPossibleMVNEEEstimatorIndex = pathTracingPath->getSegmentLength();

					//update pdf, measurement contrib and colorThroughput for BRDF-interaction:
					double dirSamplingPDF = bsdfData->getBSDFDirectionPDF(intersectionNormal, newDir, currDir);
					newMeasurementContrib = measurementContrib;
					pathTracingPDF *= dirSamplingPDF;
					newColorThroughput = colorThroughput * bsdfData->evalBSDF(intersectionNormal, newDir, currDir);
				}
				else {
					//update index for MVNEE start vertex:
					firstPossibleMVNEEEstimatorIndex = pathTracingPath->getSegmentLength() + 1;

					newMeasurementContrib = measurementContrib;
					newColorThroughput = colorThroughput;
				}

				//update variables:
				currPosition = surfaceVertex + Constants::epsilon * intersectionNormal;
				currDir = newDir;
			}
			else {
				break;
			}
		}
		else {
			//////////////////////////////
			// Medium interaction
			//////////////////////////////
			vec3 nextScatteringVertex = currPosition + (float)freePathLength * currDir;
			pathTracingPath->addHomVolVertex(nextScatteringVertex, TYPE_MEDIUM_SCATTERING);


			//update pdf and mc:
			double transmittance = exp(-medium.mu_t * freePathLength);
			double G = 1.0 / ((double)freePathLength * (double)freePathLength);
			if (!isfinite(G)) {
				break;
			}
			assert(isfinite(G));
			measurementContrib *= transmittance * G;
			pathTracingPDF *= medium.mu_t * transmittance * G;
			tl_cumulativePathTracingPDFs[pathTracingPath->getSegmentLength()] = pathTracingPDF;


			//sample direction based on Henyey-Greenstein:
			vec3 newDir = sampleHenyeyGreensteinDirection(currDir, sample1D(threadID), sample1D(threadID), medium.hg_g_F);

			//update pdf and mc for phase function
			float cos_theta_Phase = dot(currDir, newDir);
			double phase = (double)henyeyGreenstein(cos_theta_Phase, medium.hg_g_F);
			newMeasurementContrib = measurementContrib * medium.mu_s * phase;
			pathTracingPDF *= phase;

			//update variables:
			forkVertex.vertex = nextScatteringVertex;
			forkVertex.vertexType = TYPE_MEDIUM_SCATTERING;
			forkVertex.geometryID = -1;
			forkVertex.surfaceNormal = vec3(0.0f);

			currPosition = nextScatteringVertex;
			currDir = newDir;
		}

		int lastPathTracingVertexIndex = pathTracingPath->getSegmentLength();
		if (lastPathTracingVertexIndex >= rendering.MAX_SEGMENT_COUNT) {
			break;
		}

		//////////////////////////////
		// MVNEE
		//////////////////////////////		

		//only perform MVNEE on surfaces other than delta BSDFs
		if (forkVertex.vertexType != TYPE_SURFACE || (lastBSDF != 0 && !lastBSDF->isDelta())) {
			//first sample a light vertex
			int sampledLightIndex;

			bool validLightDistances = scene->calcDistancesToLights(tl_lightDistances, forkVertex.vertex);
			assert(validLightDistances);

			vec3 lightVertex = scene->sampleLightPosition(tl_lightDistances, sample1D(threadID), sample1D(threadID), sample1D(threadID), &sampledLightIndex);
			if (sampledLightIndex > -1) {
				LightSource* sampledLightSource = scene->lightSources[sampledLightIndex];

				//now sample a segment starting from the light source!
				vec3 sampledLightDir = sampledLightSource->sampleLightDirection(sample1D(threadID), sample1D(threadID));

				//intersect scene to find the maximum distance
				RTCRay lightRay;
				vec3 hitSurfaceNormal;
				scene->intersectScene(lightVertex + Constants::epsilon * sampledLightSource->normal, sampledLightDir, lightRay, hitSurfaceNormal);
				assert(lightRay.tfar > 0.0f);

				//sample distance based on transmittance between two limits
				float lightSegmentLength = sampleLimitedFreePathLength(sample1DOpenInterval(threadID), medium.mu_t, Constants::epsilon, lightRay.tfar - Constants::epsilon);

				//this is the new vertex, which MVNEE is connecting with
				vec3 mvneeDestination = lightVertex + lightSegmentLength * sampledLightDir;

				vec3 forkToDestination = mvneeDestination - forkVertex.vertex;
				float distanceToDestination = length(forkToDestination);
				if (distanceToDestination > Constants::epsilon) {

					//only perform MVNEE if the expected segment count is smaller than MAX_MVNEE_SEGMENTS
					float expectedSegments = distanceToDestination / medium.meanFreePathF;
					if (isfinite(expectedSegments) && expectedSegments > 0.0f && expectedSegments <= rendering.MAX_MVNEE_SEGMENTS_F) {

						vec3 omega = forkToDestination / distanceToDestination; //direction of the line to the light source

						vec3 u, v;
						coordinateSystem(omega, u, v); //build tangent frame: (same for every seed vertex)

						//sample the segment lengths for the seed path:
						int mvneeSegmentCount;

						bool validPath = sampleSeedSegmentLengths(distanceToDestination, tl_seedSegmentLengths, tl_seedSegmentLengthSquares, &mvneeSegmentCount, threadID);

						//special case: surface normal culling for only one segment:
						if (validPath && mvneeSegmentCount == 1) {
							if (forkVertex.vertexType == TYPE_SURFACE) {
								float cosThetaSurface = dot(omega, forkVertex.surfaceNormal);
								if (cosThetaSurface <= 0.0f) {
									validPath = false;
								}
							}
						}

						if (validPath && mvneeSegmentCount > 0) {
							//consider max segment length: since one segment is sampled always, add one extra
							if (pathTracingPath->getSegmentLength() + mvneeSegmentCount + 1 <= rendering.MAX_SEGMENT_COUNT) {

								//determine seed vertices based on sampled segment lengths:
								vec3 prevVertex = forkVertex.vertex;
								for (int s = 0; s < (mvneeSegmentCount - 1); s++) {
									vec3 newSeedVertex = prevVertex + tl_seedSegmentLengths[s] * omega;
									tl_seedVertices[s] = newSeedVertex;
									prevVertex = newSeedVertex;
								}
								//add destination vertex
								tl_seedVertices[mvneeSegmentCount - 1] = mvneeDestination;

								////////////////////////////
								//  perturb all vertices but the mvneeDestination vertex:
								////////////////////////////
								vec3 previousPerturbedVertex = forkVertex.vertex;
								double sigmaForHgSquare = sigmaForHG * sigmaForHG;

								vec3 seedVertex;
								vec3 perturbedVertex;
								bool validMVNEEPath = true;

								double leftSideFactor = 0.0;
								double rightSideFactor = 0.0;
								for (int x = 0; x < mvneeSegmentCount; x++) {
									rightSideFactor += tl_seedSegmentLengthSquares[x];
								}

								for (int p = 0; p < (mvneeSegmentCount - 1); p++) {
									seedVertex = tl_seedVertices[p];

									//sanity check: distances to start and end may not be <= 0
									float distanceToLineEnd = length(mvneeDestination - seedVertex);
									float distanceToLineStart = length(seedVertex - forkVertex.vertex);
									if (distanceToLineEnd <= 0.0f || distanceToLineStart <= 0.0f) {
										validMVNEEPath = false;
										break;
									}

									double currSegLengthSqr = tl_seedSegmentLengthSquares[p];
									leftSideFactor += currSegLengthSqr;
									rightSideFactor -= currSegLengthSqr;
									if (rightSideFactor <= 0.0f) {
										validMVNEEPath = false;
										break;
									}

									double sigma = calcGaussProductSquaredSigmas(leftSideFactor * sigmaForHgSquare, rightSideFactor * sigmaForHgSquare);
									double finalGGXAlpha = sigma * Constants::GGX_CONVERSION_Constant; //conversion with Constants factor

									//perform perturbation using ggx radius and uniform angle in u,v plane 
									perturbVertexGGX2D(finalGGXAlpha, u, v, seedVertex, &perturbedVertex, threadID);

									float maxT = tl_seedSegmentLengths[p];

									//for first perturbed vertex, check if perturbation violates the normal culling condition on the surface!
									if (p == 0 && forkVertex.vertexType == TYPE_SURFACE) {
										vec3 forkToFirstPerturbed = perturbedVertex - forkVertex.vertex;
										float firstDistToPerturbed = length(forkToFirstPerturbed);
										if (firstDistToPerturbed > 0.0f) {
											forkToFirstPerturbed /= firstDistToPerturbed;
											float cosThetaSurface = dot(forkToFirstPerturbed, forkVertex.surfaceNormal);
											if (cosThetaSurface <= 0.0f) {
												validMVNEEPath = false;
												break;
											}
										}
										else {
											validMVNEEPath = false;
											break;
										}

										previousPerturbedVertex = forkVertex.vertex + Constants::epsilon * forkVertex.surfaceNormal;

										maxT -= Constants::epsilon;
										if (maxT <= 0.0f) {
											validMVNEEPath = false;
											break;
										}
									}

									//visibility check:
									vec3 visibilityDirection = normalize(perturbedVertex - previousPerturbedVertex);

									RTCRay shadowRay;
									if (scene->vertexOccluded(previousPerturbedVertex, visibilityDirection, maxT, shadowRay)) {
										validMVNEEPath = false;
										break;
									}

									//add vertex to path
									pathTracingPath->addHomVolVertex(perturbedVertex, TYPE_MVNEE);
									previousPerturbedVertex = perturbedVertex;
								}

								if (validMVNEEPath) {
									vec3 lastSegmentDir = mvneeDestination - previousPerturbedVertex;
									float lastSegmentLength = length(lastSegmentDir);
									if (lastSegmentLength > 0.0f) {
										assert(lastSegmentLength > 0.0f);
										lastSegmentDir /= lastSegmentLength;

										//visibility check: no normal culling necessary (since we sampled the direction)
										if (mvneeSegmentCount == 1 && forkVertex.vertexType == TYPE_SURFACE) {
											previousPerturbedVertex = forkVertex.vertex + Constants::epsilon * forkVertex.surfaceNormal;

											if (lastSegmentLength - Constants::epsilon > 0.0f) {
												lastSegmentLength -= Constants::epsilon;
											}
										}

										//occlusion check:
										RTCRay lastShadowRay;
										if (!scene->vertexOccluded(previousPerturbedVertex, lastSegmentDir, lastSegmentLength, lastShadowRay)) {
											//add mvnee destination vertex and light Vertex!

											PathVertex destinationVertexStruct(mvneeDestination, TYPE_MVNEE, -1, vec3(0.0f), VERT_HOM_MEDIUM);
											pathTracingPath->addVertex(destinationVertexStruct);

											PathVertex lightVertexStruct(lightVertex, TYPE_MVNEE, -1, sampledLightSource->normal, VERT_HOM_MEDIUM);
											pathTracingPath->addVertex(lightVertexStruct);

											//calculate measurement constribution, pdf and mis weight:
											//estimator Index for MVNEE is index of last path tracing vertex
											//make sure the last path tracing vertex index is set correctly in order to use the correct PDF
											vec3 finalContribution = calcFinalWeightedContribution_LightImportanceSampling(pathTracingPath, lastPathTracingVertexIndex, sampledLightIndex, firstPossibleMVNEEEstimatorIndex, measurementContrib, colorThroughput);
											finalPixel += finalContribution;
										}

									}
									else {
										//cout << "last MVNEE segment length <= 0: " << lastSegmentLength << endl;
									}
								}
							}
							//delete all MVNEE vertices from the path construct, so path tracing stays correct
							pathTracingPath->cutMVNEEVertices();
						}

					}
				}
			}
		}
		measurementContrib = newMeasurementContrib;
		colorThroughput = newColorThroughput;
	
	}


	//Path sampling finished: now reset Path, so it can be filled again
	pathTracingPath->reset();

	return finalPixel;
}


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
inline vec3 VolumeRenderer::calcFinalWeightedContribution_LightImportanceSamplingImproved(Path* path, int estimatorIndex, int lightSourceIndex, const int firstPossibleMVNEEEstimatorIndex, const double& currentMeasurementContrib, const vec3& currentColorThroughput)
{
	const int segmentCount = path->getSegmentLength();
	assert(path->getVertexCount() >= 2);
	assert(segmentCount <= rendering.MAX_SEGMENT_COUNT);
	assert(estimatorIndex >= 0);
	assert(estimatorIndex < segmentCount);
	vec3 errorValue(0.0f); //returned in case of errors

	LightSource* hitLightSource = scene->lightSources[lightSourceIndex];

	float segmentLengthAtLight = 0.0f;
	vec3 lastDirection;
	vec3 lightVertex = path->getVertexPosition(segmentCount);

	//special case: light was immediately hit after 1 segment:
	if (path->getVertexCount() == 2) {
		vec3 firstVertex = path->getVertexPosition(0);
		vec3 dir = normalize(lightVertex - firstVertex);
		return hitLightSource->getEmissionIntensity(lightVertex, dir);
	}


	//get thread local pre-reserved data for this thread:
	int threadID = omp_get_thread_num();

	vec3* tl_seedVertices = seedVertices[threadID];
	double* tl_cumulatedPathTracingPDFs = cumulatedPathTracingPDFs[threadID];
	float* tl_seedSegmentLengths = seedSegmentLengths[threadID];
	double* tl_estimatorPDFs = estimatorPDFs[threadID];
	double* tl_estimatorPDFsLIS = estimatorPDFsLIS[threadID];
	double* tl_seedSegmentLengthSquares = seedSegmentLengthSquares[threadID];
	float* tl_lightDistances = distancesToLightSources[threadID];

	//get measurement contrib and pdf and colorThroughput for the path using path tracing up to the last path tracing vertex:
	double measurementContrib = currentMeasurementContrib;
	double pathTracingPDF = tl_cumulatedPathTracingPDFs[segmentCount];
	vec3 colorThroughput = currentColorThroughput;


	/////////////////////////////////////////////////////////////
	// calculate measurementContribution as well as path tracing pdfs
	/////////////////////////////////////////////////////////////

	//in case the path wasn't created with path tracing, measurement contrib and pdf have to be calculated for the end of the path
	if (estimatorIndex > 0) {
		pathTracingPDF = tl_cumulatedPathTracingPDFs[estimatorIndex];

		// first direction can potentially be sampled by BRDF!!
		PathVertex vertex1, vertex2;
		path->getVertex(estimatorIndex - 1, vertex1);
		path->getVertex(estimatorIndex, vertex2);

		vec3 previousDir = vertex2.vertex - vertex1.vertex;
		float previousDistance = length(previousDir);
		if (previousDistance <= 0.0f) {
			return errorValue;
		}
		previousDir /= previousDistance;

		//calculate path tracing PDF and measurement contrib for every remaining vertex:
		PathVertex currVert, prevVert;
		for (int i = estimatorIndex + 1; i < path->getVertexCount(); i++) {
			path->getVertex(i, currVert);
			path->getVertex(i - 1, prevVert);
			vec3 currentDir = currVert.vertex - prevVert.vertex;
			float currentDistance = length(currentDir);
			if (currentDistance <= 0.0f) {
				return errorValue;
			}
			currentDir = currentDir / currentDistance; //normalize direction

			double currTransmittance = exp(-medium.mu_t * (double)currentDistance);
			double currG = 1.0 / ((double)(currentDistance)* (double)(currentDistance));
			assert(isfinite(currG));

			//contribute phase function * mu_s / BRDF depending on vertex
			if (prevVert.vertexType == TYPE_SURFACE) {
				//BRDF direction
				BSDF* bsdfData = scene->getBSDF(prevVert.geometryID);

				if (!bsdfData->validOutputDirection(prevVert.surfaceNormal, currentDir)) {
					cout << "wrong outgoing direction at surface!" << endl; 
					return errorValue;
				}

				if (!bsdfData->isDelta()) {
					colorThroughput *= bsdfData->evalBSDF(prevVert.surfaceNormal, currentDir, previousDir);
					pathTracingPDF *= bsdfData->getBSDFDirectionPDF(prevVert.surfaceNormal, currentDir, previousDir);
				}
			}
			else {
				//phase direction
				float cosThetaPhase = dot(previousDir, currentDir);
				double phase = (double)henyeyGreenstein(cosThetaPhase, medium.hg_g_F);

				measurementContrib *= medium.mu_s * phase;
				pathTracingPDF *= phase;
			}

			if (i == segmentCount) {
				//Light connection segment:
				if (hitLightSource->validHitDirection(currentDir)) {
					segmentLengthAtLight = currentDistance;
					lastDirection = currentDir;

					//special treatment when light source was hit:
					float cosLightF = dot(-currentDir, hitLightSource->normal);
					assert(cosLightF >= 0.0f);
					currG *= (double)cosLightF;

					measurementContrib *= currTransmittance * currG;
					pathTracingPDF *= currTransmittance * currG;
				}
				else {
					cout << "path hits light from backside!" << endl;
					return errorValue;
				}
			}
			//contribute G-Terms, transmittance: take care of medium coefficients!
			else {
				if (currVert.vertexType == TYPE_SURFACE) {
					float cosTheta = dot(currVert.surfaceNormal, -currentDir);
					if (cosTheta <= 0.0f) {
						return errorValue;
					}
					assert(cosTheta > 0.0f);
					currG *= (double)cosTheta;

					measurementContrib *= currTransmittance * currG;
					pathTracingPDF *= currTransmittance * currG;
				}
				else {
					measurementContrib *= currTransmittance * currG;
					pathTracingPDF *= medium.mu_t * currTransmittance * currG;
				}
			}

			tl_cumulatedPathTracingPDFs[i] = pathTracingPDF;
			previousDir = currentDir;
		}
	}
	else {
		vec3 preLastVertex = path->getVertexPosition(segmentCount - 1);
		lastDirection = lightVertex - preLastVertex;
		segmentLengthAtLight = length(lastDirection);
		if (segmentLengthAtLight <= 0.0f) {
			return errorValue;
		}
		lastDirection /= segmentLengthAtLight;
	}

	//set path tracing pdf at index 0
	tl_estimatorPDFs[0] = pathTracingPDF;
	
	//////////////////////////////////////
	//calculate estimator PDFs:
	//////////////////////////////////////
	double sigmaForHgSquare = sigmaForHG * sigmaForHG;

	vec3 mvneeDestination = path->getVertexPosition(segmentCount - 1);

	//precalculate all the PDFs for the sampled light segment
	vec3 sampledLightDir = -lastDirection;

	//again trace ray to find the maximum sampling distance
	RTCRay lightRay;
	vec3 hitSurfaceNormal;
	scene->intersectScene(lightVertex + Constants::epsilon * hitLightSource->normal, sampledLightDir, lightRay, hitSurfaceNormal);

	double lightDirectionPDF = hitLightSource->getDirectionSamplingPDF(sampledLightDir) / ((double)segmentLengthAtLight * (double)segmentLengthAtLight);
	double lightDistancePDF = getLimitedFreePathPDF(segmentLengthAtLight, medium.mu_t, Constants::epsilon, lightRay.tfar);

	//there is one less MVNEE estimator, since a one segment connection is no longer possible!
	for (int e = firstPossibleMVNEEEstimatorIndex; e < (segmentCount - 1); e++) {
		double pathTracingStartPDF = tl_cumulatedPathTracingPDFs[e]; //pdf for sampling the first vertices with path tracing
		vec3 forkVertex = path->getVertexPosition(e); //last path tracing vertex and anchor point for the mvnee path

		float distanceToLight = length(lightVertex - forkVertex);

		float expectedSegments = distanceToLight / medium.meanFreePathF;

		if (distanceToLight <= 0.0f) {
			tl_estimatorPDFs[e] = 0.0;
		}
		else if (isfinite(expectedSegments) && expectedSegments > 0.0f && expectedSegments <= rendering.MAX_MVNEE_SEGMENTS_F) {
			vec3 forkToDestination = mvneeDestination - forkVertex;
			float distanceToDestination = length(forkToDestination);

			//MVNEE is only valid if expected segment count is smaller than MAX_MVNEE_SEGMENTS!			
			vec3 omega = forkToDestination / distanceToDestination; //normalize direction

			//segment count of mvnee path, which is also the mvnee vertex count (from fork vertex to mvneeDestination!)
			int mvneeSegmentCount = segmentCount - e - 1;

			//build tangent frame: (same for every seed vertex)
			vec3 u, v;
			coordinateSystem(omega, u, v);

			/////////////////////////////////
			// recover seed distances and seed vertices:
			/////////////////////////////////
			bool validMVNEEPath = true;
			float lastSeedDistance = 0.0f;
			for (int s = 0; s < (mvneeSegmentCount - 1); s++) {
				vec3 perturbedVertex = path->getVertexPosition(e + 1 + s);
				vec3 forkToPerturbed = perturbedVertex - forkVertex;

				//use orthogonal projection to recover the seed distance:
				float seedDistance = dot(forkToPerturbed, omega);

				//now some things have to be checked in order to make sure the path can be created with mvnee:
				//the new Distance from forkVertex has to be larger than the summed distance,
				//also the summedDistance may never be larger than distanceToLight!!
				if (seedDistance <= 0.0f || seedDistance <= lastSeedDistance || seedDistance >= distanceToDestination) {
					//path is invalid for mvnee, set estimatorPDF to 0
					tl_estimatorPDFs[e] = 0.0;
					validMVNEEPath = false;
					break;
				}

				assert(seedDistance > 0.0f);
				vec3 seedVertex = forkVertex + seedDistance * omega;
				tl_seedVertices[s] = seedVertex;
				float distanceToPrevVertex = seedDistance - lastSeedDistance;
				assert(distanceToPrevVertex > 0.0f);
				tl_seedSegmentLengths[s] = distanceToPrevVertex;
				tl_seedSegmentLengthSquares[s] = distanceToPrevVertex * distanceToPrevVertex;

				lastSeedDistance = seedDistance;
			}
			//last segment:
			float lastSegmentLength = distanceToDestination - lastSeedDistance;
			tl_seedSegmentLengths[mvneeSegmentCount - 1] = lastSegmentLength;
			tl_seedSegmentLengthSquares[mvneeSegmentCount - 1] = (double)lastSegmentLength * (double)lastSegmentLength;

			/////////////////////////////////
			// Calculate Perturbation PDF
			/////////////////////////////////
			double perturbationPDF = 1.0;

			double leftSideFactor = 0.0;
			double rightSideFactor = 0.0;
			for (int x = 0; x < mvneeSegmentCount; x++) {
				rightSideFactor += tl_seedSegmentLengthSquares[x];
			}

			if (validMVNEEPath) {
				//uv-perturbation for all MVNEE-vertices but the light vertex!
				for (int s = 0; s < (mvneeSegmentCount - 1); s++) {
					vec3 perturbedVertex = path->getVertexPosition(e + 1 + s);
					vec3 forkToPerturbed = perturbedVertex - forkVertex;

					double currSegLengthSqu = tl_seedSegmentLengthSquares[s];
					leftSideFactor += currSegLengthSqu;
					rightSideFactor -= currSegLengthSqu;
					if (rightSideFactor <= 0.0f) {
						validMVNEEPath = false;
						break;
					}

					double sigma = calcGaussProductSquaredSigmas(leftSideFactor * sigmaForHgSquare, rightSideFactor * sigmaForHgSquare);
					double finalGGXAlpha = sigma * Constants::GGX_CONVERSION_Constant; //multiply with Constants for conversion from gauss to ggx

					//update pdfs:
					vec3 seedToPerturbed = perturbedVertex - tl_seedVertices[s];
					float uDist = dot(seedToPerturbed, u);
					float vDist = dot(seedToPerturbed, v);
					float r2 = (uDist*uDist) + (vDist*vDist);
					double uvPerturbPDF = GGX_2D_PDF(r2, finalGGXAlpha); //pdf for u-v-plane perturbation

					float distanceToPrevVertex = tl_seedSegmentLengths[s];

					double combinedPDF = (medium.mu_t * uvPerturbPDF);
					perturbationPDF *= combinedPDF;
				}

				//set final estimator pdf if everything is valid
				if (validMVNEEPath) {
					float lastSeedSegmentLength = tl_seedSegmentLengths[mvneeSegmentCount - 1];
					if (lastSeedSegmentLength >= 0.0f) {
						perturbationPDF *= exp(-medium.mu_t * (double)distanceToDestination);

						bool validLightDistances = scene->calcDistancesToLights(tl_lightDistances, forkVertex);
						assert(validLightDistances);
						double lightVertexSamplingPDF = scene->getLightVertexSamplingPDF(tl_lightDistances, lightSourceIndex);
						double combinedLightPDF = lightVertexSamplingPDF * lightDirectionPDF * lightDistancePDF;

						double finalEstimatorPDF = pathTracingStartPDF * combinedLightPDF * perturbationPDF;
						if (isfinite(finalEstimatorPDF)) {
							tl_estimatorPDFs[e] = finalEstimatorPDF;
						}
						else {
							cout << "finalEstimatorPDF infinite!" << endl;
							tl_estimatorPDFs[e] = 0.0;
						}
					}
					else {
						tl_estimatorPDFs[e] = 0.0;
					}
				}
				else {
					tl_estimatorPDFs[e] = 0.0;
				}
			}
			else {
				tl_estimatorPDFs[e] = 0.0;
			}
		}
		else {
			tl_estimatorPDFs[e] = 0.0;
		}
	}

	/////////////////
	// NEE PDF!
	/////////////////	
	vec3 neeForkVertex = path->getVertexPosition(segmentCount - 1); //last path tracing vertex and anchor point for the mvnee path
	vec3 neeDir = lightVertex - neeForkVertex;
	tl_estimatorPDFs[segmentCount - 1] = 0.0;
	float neeDistance = length(neeDir);
	if (neeDistance > Constants::epsilon && neeDistance <= (4.0f * medium.meanFreePathF)) {
		double neePathTracingStartPDF = tl_cumulatedPathTracingPDFs[segmentCount - 1]; //pdf for sampling the first vertices with path tracing

		bool validLightDistances = scene->calcDistancesToLights(tl_lightDistances, neeForkVertex);
		assert(validLightDistances);
		tl_estimatorPDFs[segmentCount - 1] = neePathTracingStartPDF * scene->getLightVertexSamplingPDF(tl_lightDistances, lightSourceIndex);
	}

	///////////////////////////////
	// calculate MIS weight
	///////////////////////////////
	double finalPDF = tl_estimatorPDFs[estimatorIndex]; //pdf of the estimator that created the path
	if (finalPDF > 0.0) {
		float finalContribution = (float)(measurementContrib / finalPDF);
		if (isfinite(finalContribution)) {
			assert(finalContribution >= 0.0);

			//one less mvnee estimator, since the minimum segment count for MVNEE is now 2!
			int estimatorCount = segmentCount - firstPossibleMVNEEEstimatorIndex;
			float misWeight = misBalanceWeight(finalPDF, pathTracingPDF, &tl_estimatorPDFs[firstPossibleMVNEEEstimatorIndex], estimatorCount);

			vec3 finalPixel = misWeight * finalContribution * colorThroughput * hitLightSource->getEmissionIntensity(lightVertex, lastDirection);
			return finalPixel;
		}
	}

	return errorValue;
}


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
vec3 VolumeRenderer::pathTracing_MVNEE_LightImportanceSamplingImproved(const vec3& rayOrigin, const vec3& rayDir)
{
	//get thread local pre-reservers data for this thread:
	int threadID = omp_get_thread_num();
	Path* pathTracingPath = pathTracingPaths[threadID];
	vec3* tl_seedVertices = seedVertices[threadID];
	vec3* tl_perturbedVertices = perturbedVertices[threadID];
	float* tl_seedSegmentLengths = seedSegmentLengths[threadID];
	double* tl_seedSegmentLengthSquares = seedSegmentLengthSquares[threadID];
	float* tl_lightDistances = distancesToLightSources[threadID];

	double* tl_cumulativePathTracingPDFs = cumulatedPathTracingPDFs[threadID];
	tl_cumulativePathTracingPDFs[0] = 1.0;

	//add first Vertex
	PathVertex origin(rayOrigin, TYPE_ORIGIN, -1, rayDir, VERT_HOM_MEDIUM);
	pathTracingPath->addVertex(origin);

	vec3 finalPixel(0.0f);
	BSDF* lastBSDF = 0;

	vec3 currPosition = rayOrigin;
	vec3 currDir = rayDir;

	double pathTracingPDF = 1.0;
	double measurementContrib = 1.0;
	double newMeasurementContrib = 1.0;
	vec3 colorThroughput(1.0f);
	vec3 newColorThroughput(1.0f);

	//index of first vertex which could serve as a MVNEE estimator starting point
	int firstPossibleMVNEEEstimatorIndex = 1;


	while (pathTracingPath->getSegmentLength() < rendering.MAX_SEGMENT_COUNT && isNotZero(colorThroughput)) {
		//sanity check assertions:
		assert(pathTracingPath->getVertexCount() > 0);
		PathVertex lastPathVertex;
		pathTracingPath->getVertex(pathTracingPath->getSegmentLength(), lastPathVertex);
		assert(lastPathVertex.vertexType != TYPE_MVNEE);
		pathTracingPath->getVertex(pathTracingPath->getVertexCount() - 1, lastPathVertex);
		assert(lastPathVertex.vertexType != TYPE_MVNEE);

		//sample free path Lenth
		double freePathLength = sampleFreePathLength(sample1DOpenInterval(threadID), medium.mu_t);

		//this is the starting vertex for MVNEE, depending on the intersection, it can either be a surface vertex or medium vertex
		PathVertex forkVertex;

		//intersect scene
		RTCRay ray;
		vec3 intersectionNormal;
		if (scene->intersectScene(currPosition, currDir, ray, intersectionNormal) && ray.tfar <= freePathLength) {
			if (ray.tfar > Constants::epsilon) {

				//////////////////////////////
				// Surface interaction
				//////////////////////////////
				vec3 surfaceVertex = currPosition + ray.tfar * currDir;
				forkVertex.vertex = surfaceVertex;
				forkVertex.vertexType = TYPE_SURFACE;
				forkVertex.geometryID = ray.geomID;
				forkVertex.surfaceNormal = intersectionNormal;
				pathTracingPath->addVertex(forkVertex);

				int hitLightIndex;
				if (scene->lightIntersected(ray.geomID, &hitLightIndex)) {
					//////////////////////////////
					// Light hit
					//////////////////////////////
					LightSource* hitLight = scene->lightSources[hitLightIndex];

					//normal culling
					if (hitLight->validHitDirection(currDir)) {

						//update contribution
						double transmittance = exp(-medium.mu_t * ray.tfar);
						float cosThetaLight = dot(-currDir, hitLight->normal);
						double G = cosThetaLight / ((double)ray.tfar * (double)ray.tfar);
						measurementContrib *= transmittance * G;
						pathTracingPDF *= transmittance * G;
						tl_cumulativePathTracingPDFs[pathTracingPath->getSegmentLength()] = pathTracingPDF;

						//estimator Index for Path tracing is 0!
						vec3 finalContribution;
						if (pathTracingPath->getSegmentLength() > 1) {
							finalContribution = calcFinalWeightedContribution_LightImportanceSamplingImproved(pathTracingPath, 0, hitLightIndex, firstPossibleMVNEEEstimatorIndex, measurementContrib, colorThroughput);
						}
						else {
							finalContribution = hitLight->getEmissionIntensity(surfaceVertex, currDir);
						}

						finalPixel += finalContribution;
					}
					break;
				}

				//surface normal culling:
				if (dot(intersectionNormal, -currDir) <= 0.0f) {
					break;
				}


				//update pdf and mc:
				double transmittance = exp(-medium.mu_t * ray.tfar);
				float cosThetaSurface = dot(-currDir, intersectionNormal);
				double G = cosThetaSurface / ((double)ray.tfar * (double)ray.tfar);
				measurementContrib *= transmittance * G;
				pathTracingPDF *= transmittance * G;
				tl_cumulativePathTracingPDFs[pathTracingPath->getSegmentLength()] = pathTracingPDF;

				BSDF* bsdfData = scene->getBSDF(ray.geomID);
				lastBSDF = bsdfData;

				//sample direction based on BRDF:
				vec3 newDir = bsdfData->sampleBSDFDirection(intersectionNormal, currDir, sample1D(threadID), sample1D(threadID));
				if (!bsdfData->validOutputDirection(intersectionNormal, newDir)) {
					break;
				}
				if (!bsdfData->isDelta()) {

					//update index for MVNEE start vertex:
					firstPossibleMVNEEEstimatorIndex = pathTracingPath->getSegmentLength();

					//update pdf, measurement contrib and colorThroughput for BRDF-interaction:
					double dirSamplingPDF = bsdfData->getBSDFDirectionPDF(intersectionNormal, newDir, currDir);
					newMeasurementContrib = measurementContrib;
					pathTracingPDF *= dirSamplingPDF;
					newColorThroughput = colorThroughput * bsdfData->evalBSDF(intersectionNormal, newDir, currDir);
				}
				else {
					//update index for MVNEE start vertex:
					firstPossibleMVNEEEstimatorIndex = pathTracingPath->getSegmentLength() + 1;

					newMeasurementContrib = measurementContrib;
					newColorThroughput = colorThroughput;
				}

				//update variables:
				currPosition = surfaceVertex + Constants::epsilon * intersectionNormal;
				currDir = newDir;
			}
			else {
				break;
			}
		}
		else {
			//////////////////////////////
			// Medium interaction
			//////////////////////////////
			vec3 nextScatteringVertex = currPosition + (float)freePathLength * currDir;
			pathTracingPath->addHomVolVertex(nextScatteringVertex, TYPE_MEDIUM_SCATTERING);


			//update pdf and mc:
			double transmittance = exp(-medium.mu_t * freePathLength);
			double G = 1.0 / ((double)freePathLength * (double)freePathLength);
			if (!isfinite(G)) {
				break;
			}
			assert(isfinite(G));
			measurementContrib *= transmittance * G;
			pathTracingPDF *= medium.mu_t * transmittance * G;
			tl_cumulativePathTracingPDFs[pathTracingPath->getSegmentLength()] = pathTracingPDF;


			//sample direction based on Henyey-Greenstein:
			vec3 newDir = sampleHenyeyGreensteinDirection(currDir, sample1D(threadID), sample1D(threadID), medium.hg_g_F);

			//update pdf and mc for phase function
			float cos_theta_Phase = dot(currDir, newDir);
			double phase = (double)henyeyGreenstein(cos_theta_Phase, medium.hg_g_F);
			newMeasurementContrib = measurementContrib * medium.mu_s * phase;
			pathTracingPDF *= phase;

			//update variables:
			forkVertex.vertex = nextScatteringVertex;
			forkVertex.vertexType = TYPE_MEDIUM_SCATTERING;
			forkVertex.geometryID = -1;
			forkVertex.surfaceNormal = vec3(0.0f);

			currPosition = nextScatteringVertex;
			currDir = newDir;
		}

		int lastPathTracingVertexIndex = pathTracingPath->getSegmentLength();
		if (lastPathTracingVertexIndex >= rendering.MAX_SEGMENT_COUNT) {
			break;
		}

		//////////////////////////////
		// NEE / MVNEE
		//////////////////////////////		

		//only perform MVNEE/NEE on surfaces other than delta BSDFs
		if (forkVertex.vertexType != TYPE_SURFACE || (lastBSDF != 0 && !lastBSDF->isDelta())) {
			//first sample a light vertex
			int sampledLightIndex;
			bool validLightDistances = scene->calcDistancesToLights(tl_lightDistances, forkVertex.vertex);
			assert(validLightDistances);

			vec3 lightVertex = scene->sampleLightPosition(tl_lightDistances, sample1D(threadID), sample1D(threadID), sample1D(threadID), &sampledLightIndex);
			if (sampledLightIndex > -1) {
				float distanceToLight = length(lightVertex - forkVertex.vertex);
				if (distanceToLight > Constants::epsilon) {

					//only perform MVNEE if the expected segment count is smaller than MAX_MVNEE_SEGMENTS
					float expectedSegments = distanceToLight / medium.meanFreePathF;

					if (isfinite(expectedSegments) && expectedSegments > 0.0f && expectedSegments <= rendering.MAX_MVNEE_SEGMENTS_F) {
						LightSource* sampledLightSource = scene->lightSources[sampledLightIndex];

						//////////////////
						// MVNEE
						//////////////////

						//now sample a segment starting from the light source!
						vec3 sampledLightDir = sampledLightSource->sampleLightDirection(sample1D(threadID), sample1D(threadID));

						//intersect scene to find the maximum distance
						RTCRay lightRay;
						vec3 hitSurfaceNormal;
						scene->intersectScene(lightVertex + Constants::epsilon * sampledLightSource->normal, sampledLightDir, lightRay, hitSurfaceNormal);
						assert(lightRay.tfar > 0.0f);

						//sample distance based on transmittance between two limits
						float lightSegmentLength = sampleLimitedFreePathLength(sample1DOpenInterval(threadID), medium.mu_t, Constants::epsilon, lightRay.tfar);

						//this is the new vertex, which MVNEE is connecting with
						vec3 mvneeDestination = lightVertex + lightSegmentLength * sampledLightDir;

						vec3 forkToDestination = mvneeDestination - forkVertex.vertex;
						float distanceToDestination = length(forkToDestination);					

						vec3 omega = forkToDestination / distanceToDestination; //direction of the line to the light source

						vec3 u, v;
						coordinateSystem(omega, u, v); //build tangent frame: (same for every seed vertex)

						//sample the segment lengths for the seed path:
						int mvneeSegmentCount;

						bool validPath = sampleSeedSegmentLengths(distanceToDestination, tl_seedSegmentLengths, tl_seedSegmentLengthSquares, &mvneeSegmentCount, threadID);

						//special case: surface normal culling for only one segment:
						if (validPath && mvneeSegmentCount == 1) {
							if (forkVertex.vertexType == TYPE_SURFACE) {
								float cosThetaSurface = dot(omega, forkVertex.surfaceNormal);
								if (cosThetaSurface <= 0.0f) {
									validPath = false;
								}
							}
						}

						if (validPath && mvneeSegmentCount > 0) {
							//consider max segment length: since one segment is sampled always, add one extra
							if (pathTracingPath->getSegmentLength() + mvneeSegmentCount + 1 <= rendering.MAX_SEGMENT_COUNT) {

								//determine seed vertices based on sampled segment lengths:
								vec3 prevVertex = forkVertex.vertex;
								for (int s = 0; s < (mvneeSegmentCount - 1); s++) {
									vec3 newSeedVertex = prevVertex + tl_seedSegmentLengths[s] * omega;
									tl_seedVertices[s] = newSeedVertex;
									prevVertex = newSeedVertex;
								}
								//add destination vertex
								tl_seedVertices[mvneeSegmentCount - 1] = mvneeDestination;

								////////////////////////////
								//  perturb all vertices but the mvneeDestination vertex:
								////////////////////////////
								vec3 previousPerturbedVertex = forkVertex.vertex;
								double sigmaForHgSquare = sigmaForHG * sigmaForHG;

								vec3 seedVertex;
								vec3 perturbedVertex;
								bool validMVNEEPath = true;

								double leftSideFactor = 0.0;
								double rightSideFactor = 0.0;
								for (int x = 0; x < mvneeSegmentCount; x++) {
									rightSideFactor += tl_seedSegmentLengthSquares[x];
								}

								for (int p = 0; p < (mvneeSegmentCount - 1); p++) {
									seedVertex = tl_seedVertices[p];

									//sanity check: distances to start and end may not be <= 0
									float distanceToLineEnd = length(mvneeDestination - seedVertex);
									float distanceToLineStart = length(seedVertex - forkVertex.vertex);
									if (distanceToLineEnd <= 0.0f || distanceToLineStart <= 0.0f) {
										validMVNEEPath = false;
										break;
									}

									double currSegLengthSqr = tl_seedSegmentLengthSquares[p];
									leftSideFactor += currSegLengthSqr;
									rightSideFactor -= currSegLengthSqr;
									if (rightSideFactor <= 0.0f) {
										validMVNEEPath = false;
										break;
									}

									double sigma = calcGaussProductSquaredSigmas(leftSideFactor * sigmaForHgSquare, rightSideFactor * sigmaForHgSquare);
									double finalGGXAlpha = sigma * Constants::GGX_CONVERSION_Constant; //conversion with Constants factor

									//perform perturbation using ggx radius and uniform angle in u,v plane 
									perturbVertexGGX2D(finalGGXAlpha, u, v, seedVertex, &perturbedVertex, threadID);

									float maxT = tl_seedSegmentLengths[p];

									//for first perturbed vertex, check if perturbation violates the normal culling condition on the surface!
									if (p == 0 && forkVertex.vertexType == TYPE_SURFACE) {
										vec3 forkToFirstPerturbed = perturbedVertex - forkVertex.vertex;
										float firstDistToPerturbed = length(forkToFirstPerturbed);
										if (firstDistToPerturbed > 0.0f) {
											forkToFirstPerturbed /= firstDistToPerturbed;
											float cosThetaSurface = dot(forkToFirstPerturbed, forkVertex.surfaceNormal);
											if (cosThetaSurface <= 0.0f) {
												validMVNEEPath = false;
												break;
											}
										}
										else {
											validMVNEEPath = false;
											break;
										}

										previousPerturbedVertex = forkVertex.vertex + Constants::epsilon * forkVertex.surfaceNormal;

										maxT -= Constants::epsilon;
										if (maxT <= 0.0f) {
											validMVNEEPath = false;
											break;
										}
									}

									//visibility check:

									vec3 visibilityDirection = normalize(perturbedVertex - previousPerturbedVertex);

									RTCRay shadowRay;
									if (scene->vertexOccluded(previousPerturbedVertex, visibilityDirection, maxT, shadowRay)) {
										validMVNEEPath = false;
										break;
									}

									//add vertex to path
									pathTracingPath->addHomVolVertex(perturbedVertex, TYPE_MVNEE);
									previousPerturbedVertex = perturbedVertex;
								}

								if (validMVNEEPath) {
									vec3 lastSegmentDir = mvneeDestination - previousPerturbedVertex;
									float lastSegmentLength = length(lastSegmentDir);
									if (lastSegmentLength > 0.0f) {
										assert(lastSegmentLength > 0.0f);
										lastSegmentDir /= lastSegmentLength;

										//visibility check: no normal culling necessary (since we sampled the direction)
										if (mvneeSegmentCount == 1 && forkVertex.vertexType == TYPE_SURFACE) {
											previousPerturbedVertex = forkVertex.vertex + Constants::epsilon * forkVertex.surfaceNormal;

											if (lastSegmentLength - Constants::epsilon > 0.0f) {
												lastSegmentLength -= Constants::epsilon;
											}
										}

										//occlusion check:
										RTCRay lastShadowRay;
										if (!scene->vertexOccluded(previousPerturbedVertex, lastSegmentDir, lastSegmentLength, lastShadowRay)) {
											//add mvnee destination vertex and light Vertex!

											PathVertex destinationVertexStruct(mvneeDestination, TYPE_MVNEE, -1, vec3(0.0f), VERT_HOM_MEDIUM);
											pathTracingPath->addVertex(destinationVertexStruct);

											PathVertex lightVertexStruct(lightVertex, TYPE_MVNEE, -1, sampledLightSource->normal, VERT_HOM_MEDIUM);
											pathTracingPath->addVertex(lightVertexStruct);

											//calculate measurement constribution, pdf and mis weight:
											//estimator Index for MVNEE is index of last path tracing vertex
											//make sure the last path tracing vertex index is set correctly in order to use the correct PDF
											vec3 finalContribution = calcFinalWeightedContribution_LightImportanceSamplingImproved(pathTracingPath, lastPathTracingVertexIndex, sampledLightIndex, firstPossibleMVNEEEstimatorIndex, measurementContrib, colorThroughput);
											finalPixel += finalContribution;
										}

									}
									else {
										//cout << "last MVNEE segment length <= 0: " << lastSegmentLength << endl;
									}
								}
							}
							//delete all MVNEE vertices from the path construct, so path tracing stays correct
							pathTracingPath->cutMVNEEVertices();
						}

						/////////////////////
						// NEE
						/////////////////////

						//only perform NEE whenever the distance to the light source is reasonably small:
						if (expectedSegments <= (4.0f * medium.meanFreePathF)) {
							vec3 forkToLight = normalize(lightVertex - forkVertex.vertex);

							//surface normal culling:
							if (forkVertex.vertexType != TYPE_SURFACE || dot(forkVertex.surfaceNormal, forkToLight) > 0.0f) {
								//light normal culling
								if (sampledLightSource->validHitDirection(forkToLight)) {

									//occlusion check:
									vec3 startPos = forkVertex.vertex;
									if (forkVertex.vertexType == TYPE_SURFACE) {
										startPos += Constants::epsilon * forkVertex.surfaceNormal;
										distanceToLight -= Constants::epsilon;
									}
									RTCRay shadowRay;
									if (!scene->vertexOccluded(startPos, forkToLight, distanceToLight, shadowRay)) {

										PathVertex lightVertexStruct(lightVertex, TYPE_MVNEE, -1, sampledLightSource->normal, VERT_HOM_MEDIUM);
										pathTracingPath->addVertex(lightVertexStruct);

										//calculate measurement constribution, pdf and mis weight:
										//estimator Index for MVNEE is index of last path tracing vertex
										//make sure the last path tracing vertex index is set correctly in order to use the correct PDF
										vec3 finalContribution = calcFinalWeightedContribution_LightImportanceSamplingImproved(pathTracingPath, lastPathTracingVertexIndex, sampledLightIndex, firstPossibleMVNEEEstimatorIndex, measurementContrib, colorThroughput);
										finalPixel += finalContribution;

										//delete all MVNEE vertices from the path construct, so path tracing stays correct
										pathTracingPath->cutMVNEEVertices();
									}
								}
							}
						}
					}
				}
			}
		}
		measurementContrib = newMeasurementContrib;
		colorThroughput = newColorThroughput;
	}

	//Path sampling finished: now reset Path, so it can be filled again
	pathTracingPath->reset();

	return finalPixel;
}


/**
* Constructs a MVNEE connection between start and end vertex. All vertices but the start vertex are added to the path object. Visibility checks are performed.
* Light normal culling is not done, since the end vertex might not be a light vertex. In case creation is not possible , false is returned, otherwise true.
*/
bool VolumeRenderer::constructMVNEEConnection(const PathVertex& startVertex, const vec3& endVertex, Path* path, const int threadID, int* mvneeSegmentCountOut)
{
	vec3* tl_seedVertices = seedVertices[threadID];
	vec3* tl_perturbedVertices = perturbedVertices[threadID];
	float* tl_seedSegmentLengths = seedSegmentLengths[threadID];
	double* tl_seedSegmentLengthSquares = seedSegmentLengthSquares[threadID];

	vec3 omega = endVertex - startVertex.vertex;
	float distanceToEnd = length(omega);
	if (distanceToEnd < Constants::epsilon) {
		return false;
	}
	omega /= distanceToEnd; //direction of the line to the end

	vec3 u, v;
	coordinateSystem(omega, u, v); //build tangent frame: (same for every seed vertex)

	//sample the segment lengths for the seed path:
	int mvneeSegmentCount;
	bool validPath = sampleSeedSegmentLengths(distanceToEnd, tl_seedSegmentLengths, tl_seedSegmentLengthSquares, &mvneeSegmentCount, threadID);
	if (!validPath) {
		return false;
	}
	if (mvneeSegmentCount == 1) {
		if (startVertex.vertexType == TYPE_SURFACE) {
			float cosThetaSurface = dot(omega, startVertex.surfaceNormal);
			if (cosThetaSurface <= 0.0f) {
				return false;
			}
		}
	}

	//consider max segment length:
	if (path->getSegmentLength() + mvneeSegmentCount <= rendering.MAX_SEGMENT_COUNT) {

		//determine seed vertices based on sampled segment lengths:
		vec3 prevVertex = startVertex.vertex;
		for (int s = 0; s < (mvneeSegmentCount - 1); s++) {
			vec3 newSeedVertex = prevVertex + tl_seedSegmentLengths[s] * omega;
			tl_seedVertices[s] = newSeedVertex;
			prevVertex = newSeedVertex;
		}
		//add end vertex
		tl_seedVertices[mvneeSegmentCount - 1] = endVertex;

		////////////////////////////
		//  perturb all vertices but the end vertex:
		////////////////////////////
		vec3 previousPerturbedVertex = startVertex.vertex;
		double sigmaForHgSquare = sigmaForHG * sigmaForHG;

		vec3 seedVertex;
		vec3 perturbedVertex;
		bool validMVNEEPath = true;

		double leftSideFactor = 0.0;
		double rightSideFactor = 0.0;
		for (int x = 0; x < mvneeSegmentCount; x++) {
			rightSideFactor += tl_seedSegmentLengthSquares[x];
		}

		for (int p = 0; p < (mvneeSegmentCount - 1); p++) {
			seedVertex = tl_seedVertices[p];

			//sanity check: distances to start and end may not be <= 0
			float distanceToLineEnd = length(endVertex - seedVertex);
			float distanceToLineStart = length(seedVertex - startVertex.vertex);
			if (distanceToLineEnd <= 0.0f || distanceToLineStart <= 0.0f) {
				path->cutMVNEEVertices();
				return false;
			}

			double currSegLengthSqr = tl_seedSegmentLengthSquares[p];
			leftSideFactor += currSegLengthSqr;
			rightSideFactor -= currSegLengthSqr;
			if (rightSideFactor <= 0.0f) {
				path->cutMVNEEVertices();
				return false;
			}

			double sigma = calcGaussProductSquaredSigmas(leftSideFactor * sigmaForHgSquare, rightSideFactor * sigmaForHgSquare);
			double finalGGXAlpha = sigma * Constants::GGX_CONVERSION_Constant; //conversion with Constants factor

			//perform perturbation using ggx radius and uniform angle in u,v plane 
			perturbVertexGGX2D(finalGGXAlpha, u, v, seedVertex, &perturbedVertex, threadID);

			float maxT = tl_seedSegmentLengths[p];

			//for first perturbed vertex, check if perturbation violates the normal culling condition on the surface!
			if (p == 0 && startVertex.vertexType == TYPE_SURFACE) {
				vec3 forkToFirstPerturbed = perturbedVertex - startVertex.vertex;
				float firstDistToPerturbed = length(forkToFirstPerturbed);
				if (firstDistToPerturbed > 0.0f) {
					forkToFirstPerturbed /= firstDistToPerturbed;
					float cosThetaSurface = dot(forkToFirstPerturbed, startVertex.surfaceNormal);
					if (cosThetaSurface <= 0.0f) {
						return false;
					}
				}
				else {
					return false;
				}

				previousPerturbedVertex = startVertex.vertex + Constants::epsilon * startVertex.surfaceNormal;

				maxT -= Constants::epsilon;
				if (maxT <= 0.0f) {
					return false;
				}
			}

			//visibility check:
			vec3 visibilityDirection = normalize(perturbedVertex - previousPerturbedVertex);

			RTCRay shadowRay;
			if (scene->vertexOccluded(previousPerturbedVertex, visibilityDirection, maxT, shadowRay)) {
				path->cutMVNEEVertices();
				return false;
			}

			//add vertex to path
			path->addHomVolVertex(perturbedVertex, TYPE_MVNEE);
			previousPerturbedVertex = perturbedVertex;
		}

		
		vec3 lastSegmentDir = endVertex - previousPerturbedVertex;
		float lastSegmentLength = length(lastSegmentDir);
		if (lastSegmentLength > Constants::epsilon) {
			assert(lastSegmentLength > 0.0f);
			lastSegmentDir /= lastSegmentLength;

			if (mvneeSegmentCount == 1 && startVertex.vertexType == TYPE_SURFACE) {
				previousPerturbedVertex = startVertex.vertex + Constants::epsilon * startVertex.surfaceNormal;
				if (lastSegmentLength - Constants::epsilon > 0.0f) {
					lastSegmentLength -= Constants::epsilon;
				}
			}

			//occlusion check:
			RTCRay lastShadowRay;
			if (!scene->vertexOccluded(previousPerturbedVertex, lastSegmentDir, lastSegmentLength, lastShadowRay)) {
				PathVertex endVertexStruct(endVertex, TYPE_MVNEE, -1, vec3(0.0f), VERT_HOM_MEDIUM);
				path->addVertex(endVertexStruct);

				*mvneeSegmentCountOut = mvneeSegmentCount;

				return true;
			}			
		}		
	}
	path->cutMVNEEVertices();
	return false;
}


/** Perturb a vertex on the tangent plane using a ggx sampled radius, and a uniformly sampled angle
* @param input: vector that has to be perturbed
* @param alpha width of ggx bell curve
* @param u: first tangent vector as perturbation direction 1
* @param v: second tangent vector as perturbation direction 2
* @return output: perturbed input vector
*/
inline void VolumeRenderer::perturbVertexGGX2D(const double& alpha, const vec3& u, const vec3& v, const vec3& input, vec3* output, const int threadID)
{
	glm::vec2 uv = sampleGGX2D(alpha, sample1D(threadID), sample1D(threadID));
	vec3 out = input + uv.x * u + uv.y * v;

	*output = out;
}


/**
* Samples the appropriate segment count. This is done by summing up sampled free path lengths (transmittance formula as in path tracing)
* until the curve_length is exceeded. Also returns the segments in an array (maximal MAX_PATH_LENGTH).
* Return value is false if MAX_SEGMENT_LENGTH is exceeded, true otherwise.
*/
bool VolumeRenderer::sampleSeedSegmentLengths(const double& curve_length, float* segments, int* segmentCount, const int threadID)
{
	assert(curve_length > 0.0);
	double lengthSum = 0.0;
	int vertexCount = 0;
	bool valid = true;
	while (lengthSum < curve_length) {
		if (vertexCount >= rendering.MAX_SEGMENT_COUNT) {
			valid = false;
			break;
		}

		double currSegmentLength = sampleFreePathLength(sample1DOpenInterval(threadID), medium.mu_t);
		if (lengthSum + currSegmentLength <= curve_length)
		{
			segments[vertexCount] = (float)currSegmentLength;
		}
		else {
			segments[vertexCount] = (float)(curve_length - lengthSum);
		}
		lengthSum += currSegmentLength;
		vertexCount++;
	}

	*segmentCount = vertexCount;
	return valid;
}

/**
* Samples the appropriate segment count. This is done by summing up sampled free path lengths (transmittance formula as in path tracing)
* until the curve_length is exceeded. Also returns the segments in an array (maximal MAX_PATH_LENGTH) as well as the squared segment lengths.
* Return value is false if MAX_SEGMENT_LENGTH is exceeded, true otherwise.
*/
inline bool VolumeRenderer::sampleSeedSegmentLengths(const double& curve_length, float* segments, double* segmentSquares, int* segmentCount, const int threadID)
{
	assert(curve_length > 0.0);
	double lengthSum = 0.0;
	int vertexCount = 0;
	bool valid = true;
	while (lengthSum < curve_length) {
		if (vertexCount >= rendering.MAX_SEGMENT_COUNT) {
			valid = false;
			break;
		}

		double currSegmentLength = sampleFreePathLength(sample1DOpenInterval(threadID), medium.mu_t);
		if (lengthSum + currSegmentLength <= curve_length)
		{
			segments[vertexCount] = (float)currSegmentLength;
			segmentSquares[vertexCount] = currSegmentLength * currSegmentLength;
		}
		else {
			double lastSegmentL = curve_length - lengthSum;
			segments[vertexCount] = (float)(lastSegmentL);
			segmentSquares[vertexCount] = lastSegmentL * lastSegmentL;
		}
		lengthSum += currSegmentLength;
		vertexCount++;
	}

	*segmentCount = vertexCount;
	return valid;
}

/**
* Samples the appropriate segment count. This is done by summing up sampled free path lengths (transmittance formula as in path tracing)
* until the curve_length is exceeded. Also returns the segments in an array (maximal MAX_PATH_LENGTH) as well as the squared segment lengths.
* The muT factor is given as parameter!
* Return value is false if MAX_SEGMENT_LENGTH is exceeded, true otherwise.
*/
inline bool VolumeRenderer::sampleSeedSegmentLengths(const double& curve_length, double muT, float* segments, double* segmentSquares, int* segmentCount, const int threadID)
{
	assert(curve_length > 0.0);
	double lengthSum = 0.0;
	int vertexCount = 0;
	bool valid = true;
	while (lengthSum < curve_length) {
		if (vertexCount >= rendering.MAX_SEGMENT_COUNT) {
			valid = false;
			break;
		}

		double currSegmentLength = sampleFreePathLength(sample1DOpenInterval(threadID), muT);
		if (lengthSum + currSegmentLength <= curve_length)
		{
			segments[vertexCount] = (float)currSegmentLength;
			segmentSquares[vertexCount] = currSegmentLength * currSegmentLength;
		}
		else {
			double lastSegmentL = curve_length - lengthSum;
			segments[vertexCount] = (float)(lastSegmentL);
			segmentSquares[vertexCount] = lastSegmentL * lastSegmentL;
		}
		lengthSum += currSegmentLength;
		vertexCount++;
	}

	*segmentCount = vertexCount;
	return valid;
}


void VolumeRenderer::saveBufferToTGA(const char* filename, vec3* imageBuffer, int imageWidth, int imageHeight)
{
	ofstream o(filename, ios::out | ios::binary);
	short width = (short)imageWidth;
	short height = (short)imageHeight;

	//Write the header
	o.put(0);
	o.put(0);
	o.put(2);                         /* uncompressed RGB */
	o.put(0); 		o.put(0);
	o.put(0); 	o.put(0);
	o.put(0);
	o.put(0); 	o.put(0);           /* X origin */
	o.put(0); 	o.put(0);           /* y origin */
	o.put((width & 0x00FF));
	o.put((width & 0xFF00) / 256);
	o.put((height & 0x00FF));
	o.put((height & 0xFF00) / 256);
	o.put(32);                        /* 24 bit bitmap */
	o.put(0);

	//Write the pixel data
	const unsigned char a = 255;
	for (int y = 0; y < imageHeight; y++) {
		for (int x = 0; x < imageWidth; x++) {
			vec3 currentPixel = imageBuffer[x + imageWidth * y];
			//vec3 currentPixel = frameBuffer[x][y];

			/*float gamma_corrected_r = currentPixel.r;
			float gamma_corrected_g = currentPixel.g;
			float gamma_corrected_b = currentPixel.b;*/

			//fake tonemap
			/*float gamma_corrected_r = currentPixel.r / (currentPixel.r + 1.0f);
			float gamma_corrected_g = currentPixel.g / (currentPixel.g + 1.0f);
			float gamma_corrected_b = currentPixel.b / (currentPixel.b + 1.0f);*/

			float gamma_corrected_r = powf(currentPixel.r, 1.0f/2.2f);
			float gamma_corrected_g = powf(currentPixel.g, 1.0f/2.2f);
			float gamma_corrected_b = powf(currentPixel.b, 1.0f/2.2f);

			//cap float value so that it fits into char
			//WARNING: tone mapping?
			const unsigned char r = (unsigned char)(fminf(gamma_corrected_r * 255.0f, 255.0f));
			const unsigned char g = (unsigned char)(fminf(gamma_corrected_g * 255.0f, 255.0f));
			const unsigned char b = (unsigned char)(fminf(gamma_corrected_b * 255.0f, 255.0f));

			o.put(b);
			o.put(g);
			o.put(r);
			o.put(a);
		}
	}

	//close the file
	o.close();
}

void VolumeRenderer::printRenderingParameters(int sampleCount, double duration)
{
	//save textual output of all the parameters (volume, samples, lamp,...)
	string paramsFile = rendering.sessionName;
	paramsFile.append("_params.txt");
	ofstream o(paramsFile.c_str(), ios::out);

	o << "Integrator: " << endl;
	switch (rendering.integrator) {
		case PATH_TRACING_RANDOM_WALK: o << "PATH_TRACING_RANDOM_WALK" << endl; break;
		case PATH_TRACING_NEE_MIS: o << "PATH_TRACING_NEE_MIS" << endl; break;
		case PATH_TRACING_MVNEE: o << "PATH_TRACING_MVNEE" << endl; break;
		case PATH_TRACING_MVNEE_FINAL: o << "PATH_TRACING_MVNEE_FINAL" << endl; break;
		case PATH_TRACING_MVNEE_LIGHT_IMPORTANCE_SAMPLING: o << "PATH_TRACING_MVNEE_LIGHT_IMPORTANCE_SAMPLING" << endl; break;
		case PATH_TRACING_MVNEE_LIGHT_IMPORTANCE_SAMPLING_IMPROVED: o << "PATH_TRACING_MVNEE_LIGHT_IMPORTANCE_SAMPLING_IMPROVED" << endl; break;
		default: o << "DEFAULT: Integrator unknown!" << endl; break;
	}
	o << endl;
	o << "Image Dimension: width = " << rendering.WIDTH << ", height = " << rendering.HEIGHT << endl;
	o << "Num Samples: " << sampleCount << endl;
	o << "maximum path length: " << rendering.MAX_SEGMENT_COUNT << endl;
	o << "maximum expected MVNEE segments: " << rendering.MAX_MVNEE_SEGMENTS << endl;
	o << "light choice sampling: "; scene->printLightChoiceStrategy(o);
	o << endl;
	for (int i = 0; i < scene->lightSourceCount; i++) {
		scene->lightSources[i]->printParameters(o);
		o << endl;
	}
	o << "mu_s: " << medium.mu_s << endl;
	o << "mu_a: " << medium.mu_a << endl;
	o << "mu_t: " << medium.mu_t << endl;
	o << "henyey-greenstein g: " << medium.hg_g << endl;
	o << "sigma_for_hg: " << sigmaForHG << endl;
	o << endl;
	o << "Thread count for execution = ";
	if (!rendering.RENDER_PARALLEL || rendering.THREAD_COUNT == 1) {
		o << 1 << endl;
	}
	else {
		o << rendering.THREAD_COUNT << endl;
	}	
	o << "Time elapsed: " << duration << endl; 

	//close the file
	o.close();
}


void VolumeRenderer::writeBufferToFloatFile(const string& fileName, int width, int height, vec3* buffer) {
	string fileN = fileName;
	fileN.append(".flt");

	ofstream o;
	o.open(fileN.c_str(), ios::out | ios::binary);
	if (o.good()) {
		//first write width and height in int
		o.write((char*)&width, sizeof(int));
		o.write((char*)&height, sizeof(int));
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int index = y * width + x;
				float xValue = buffer[index].x;
				o.write((char*)&xValue, sizeof(float));

				float yValue = buffer[index].y;
				o.write((char*)&yValue, sizeof(float));

				float zValue = buffer[index].z;
				o.write((char*)&zValue, sizeof(float));
			}
		}

		o.close();
	}
	else {
		cout << "error: couldnt open file!" << endl;
	}

}


vec3* VolumeRenderer::readFloatFileToBuffer(const string& fileName, int* width, int* height) {
	ifstream i;
	i.open(fileName.c_str(), ios::in | ios::binary);
	if (i.good()) {
		//read width and height
		int w;
		int h;
		i.read((char*)&w, sizeof(int));
		i.read((char*)&h, sizeof(int));
		*width = w;
		*height = h;

		vec3* buffer = new vec3[w * h];

		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				int index = y * w + x;
				float xValue, yValue, zValue;
				i.read((char*)&xValue, sizeof(float));
				i.read((char*)&yValue, sizeof(float));
				i.read((char*)&zValue, sizeof(float));
				buffer[index] = vec3(xValue, yValue, zValue);
			}
		}

		i.close();
		return buffer;
	}
	else {
		cout << "error: couldnt open file!" << endl;
	}
	return 0;
}


/** Calculate PDF for sampling the given value with a gaussian with given standard deviation  */
double VolumeRenderer::gaussPDF(const float& x, const double& stdDeviation)
{
	double exponent = (((double)(x)*(double)(x)) / (2.0 * stdDeviation * stdDeviation));
	return (1.0 / (sqrt(2.0 * M_PI) * stdDeviation)) * exp(-exponent);
}





