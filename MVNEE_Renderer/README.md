------------------------
VolumeRenderer for Multiple Vertex Next Event Estimation:

This is a Volume Rendering Tool for simple scenes in participating media, 
featuring Random Walk Path Tracing, NEE and Multiple Vertex Next Event Estimation in different versions.
It is a subset of the code used for the Paper "Multiple Vertex Next Event Estimation". The code for
heterogeneous media can't be published, but everything else is provided in this repository.


------------------------
Installation Guide
------------------------
			
	Linux:
		For Linux, a makefile is provided in the "VolumeRendererV2" folder. Just run the makefile with "make". The embree library
		used for object intersections will be used automatically. Note that this makefile only works for 64 bit systems.
		
	Windows:
		For Windows 7 to Windows 10, a 64 bit Visual Studio .sln Solution file is provided. The necessary libraries are included
		with relative paths already, the dlls are provided as well. Copy the dlls to the executable loaction. When using a 64 bit run time environment,
		the programm should be able to start immediately using the sln file. Make sure OpenMP support is acitivated in Visual
		Studio project settings.
		
		Make sure the embree library is linked, see /include and /lib folders.
		
	Error cases:
		In cases of errors, try compiling Embree for your own operating system and include it for compilation.
		The code is very simple, so you should be able to adjust it to your own needs.
		
	Running the VolumeRenderer:
		In order to run the Renderering Process, simply start the compiled executable. This will load the default scene configuration.
		Other scene files can be used by setting the scene file path as the run time argument. Example on linux:
		
		./volumerenderer ../setups/myScene.xml
		
------------------------
Code Explanation
------------------------

	The entire code is written in C++ 11, all libraries are only provided in 64 bit version. 

	The scenes are specified in xml format, with a default scene provided in "/setups/default.xml". 	
	Models can be loaded from .obj files. These objects have to be specified in the scene file as well, make sure the path to the objects is 
	specified relative to the location of the source code!
	In the integrator settings of the Scene file, make sure to adjust the thread count to your machine settings. 
	The scene file specifies parameters that can be looked up in the Settings.h file of the source code, in some cases in the Scene.h file.
	
	VolumeRenderer:
	
		The VolumeRenderer class contains the rendering code for ray generation, framebuffer and integrators. All integrators are specified as functions with additional helper functions inside
		this class. Following integrators are provided:
		
			PATH_TRACING_NO_SCATTERING: 
				Standard path tracing, excluding medium interaction.
				
			PATH_TRACING_NEE_MIS_NO_SCATTERING:
				Standard path tracing combined with Next Event Estimation, excluding medium interaction.
				
			PATH_TRACING_RANDOM_WALK:
				Random Walk Path Tracing implementation for Multiple Scattering in homogeneous media.
				
			PATH_TRACING_NEE_MIS:
				Random Walk Path Tracing combined with Next Event Estimation for Multiple Scattering in homogeneous media.
				
			PATH_TRACING_MVNEE:
				First version of combination of Random Walk Path Tracing with Multiple Vertex Next Event Estimation. This version
				is rather slow, yet relatively easy to understand. 
				
			PATH_TRACING_MVNEE_FINAL:
				Optimized version of combination of Random Walk Path Tracing with Multiple Vertex Next Event Estimation. This version
				is faster, due to some improvements.  
				
			PATH_TRACING_MVNEE_LIGHT_IMPORTANCE_SAMPLING:
				MVNEE adjusted for light source directions
				
			PATH_TRACING_MVNEE_LIGHT_IMPORTANCE_SAMPLING_IMPROVED: 
				MVNEE adjusted for light source directions, additionally NEE as third estimator
				