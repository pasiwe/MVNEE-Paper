#pragma once

#include <string>
#include <sstream>
#include "glm/glm.hpp"
#include <math.h>
#include <vector>

using namespace std;
using namespace glm;

//Decorator for std::string, adds some useful functions when decorating, can be "undecorated" by calling getStdStringData()
class StringParser {
private:
	string stringData;
public:
	StringParser();
	StringParser(string stringData);
	~StringParser();

	const char* getCharacterData();
	string* getStdString();
	string getStdStringData();

	bool startsWith(string compareString);
	bool equals(string compareString);
	bool startsWith(StringParser compareString);
	bool equals(StringParser compareString);

	int lastIndexOf(string compareString);

	/**
	* For a file path stored in this object, i.e. "C://folder1//folder2//file.txt" this function
	* erases the filename to leave the path to the folder only -> "C://folder1//folder2//"
	*/
	StringParser eraseFileNameFromFilePath();

	/**
	* For a file path stored in this object, i.e. "C://folder1//folder2//file.txt" this function
	* extracts the filename -> "file.txt"
	*/
	StringParser extractFileNameFromFilePath();

	int size();

	void append(char c);

	char at(int index);

	/**
	* This function returns the argument of a string of the format: "param argument"
	* i.e. for a line in a file: "para abc xyz" retrieveArgumentForParam("para") would return "abc xyz"
	*/
	string retrieveArgumentForParam(string paramName);

	/**
	* This function returns the arguments of a string of the format: "param argument"
	* i.e. for a line in a file: "Ns 0.2" getFloatParam("Ns") would return 0.2f
	*/
	float getFloatParam(string paramName);

	/**
	* This function returns the arguments of a string of the format: "param argument"
	* i.e. for a line in a file: "Ns 0.2" getDoubleParam("Ns") would return 0.2
	*/
	double getDoubleParam(string paramName);

	/**
	* This function returns the arguments of a string of the format: "param argument"
	* i.e. for a line in a file: "illumination 3" getIntParam("illumination") would return 3
	*/
	int getIntParam(string paramName);

	/**
	* This function returns the arguments of a string of the format: "param argument"
	* i.e. for a line in a file: "Kd 0.2 0.2 0.3" getVec3Param("Kd") would return vec3(0.2, 0.2, 0.3)
	*/
	vec3 getVec3Param(string paramName);

	/**
	* This function returns the arguments of a string of the format: "param argument"
	* i.e. for a line in a file: "face 1 2 3" getVec3Param("face") would return ivec3(1, 2, 3)
	*/
	ivec3 getIVec3Param(string paramName);

	vec2 getVec2Param(string paramName);

	vec2 getTexCoordParam(string paramName);


	void append(StringParser newEnd);

	// "==" Operator überschreiben:
	bool operator== (StringParser const& rhs);
	bool operator== (string const& rhs);

	// "!=" Operator überschreiben:
	bool operator!= (StringParser const& rhs);
	bool operator!= (string const& rhs);

	void split(char seperator, vector<StringParser>* resultStrings);
};