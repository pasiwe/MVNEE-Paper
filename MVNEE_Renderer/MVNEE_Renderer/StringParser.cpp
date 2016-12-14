#include "StringParser.h"

StringParser::StringParser()
{
	stringData = "";
}

StringParser::StringParser(string stringData) : stringData(stringData)
{

}

StringParser::~StringParser()
{

}

int StringParser::size()
{
	return (int) stringData.size();
}

char StringParser::at(int index)
{
	return stringData.at(index);
}

const char* StringParser::getCharacterData()
{
	return stringData.c_str();
}

void StringParser::append(char c)
{
	string s_c = string(1, c);
	stringData.append(s_c);
}

string* StringParser::getStdString()
{
	return &stringData;
}

string StringParser::getStdStringData()
{
	return stringData;
}

bool StringParser::startsWith(string compareString)
{
	bool result = true;

	if (stringData.size() == 0) {
		return false;
	}

	//sanity check if compareString is longer than original
	if (compareString.size() > stringData.size()) {
		compareString = compareString.substr(0, stringData.size());
	}

	int i = 0;
	while (i < compareString.size()) {
		if (stringData.at(i) != compareString.at(i)) {
			result = false;
			break;
		}
		i++;
	}

	return result;
}

bool StringParser::equals(string compareString)
{
	bool result = true;

	//sanity check if compareString is longer than original
	if (compareString.size() != stringData.size()) {
		result = false;
	}
	else {
		int i = 0;
		while (i < compareString.size()) {
			if (stringData.at(i) != compareString.at(i)) {
				result = false;
				break;
			}
			i++;
		}
	}
	return result;
}

bool StringParser::startsWith(StringParser compareString)
{
	return startsWith(compareString.getStdStringData());
}

bool StringParser::equals(StringParser compareString)
{
	return equals(compareString.getStdStringData());
}

bool StringParser::operator== (StringParser const& rhs)
{
	return equals(rhs);
}

bool StringParser::operator== (string const& rhs)
{
	return equals(rhs);
}

bool StringParser::operator!= (StringParser const& rhs)
{
	return !equals(rhs);
}

bool StringParser::operator!= (string const& rhs)
{
	return !equals(rhs);
}

/**
*finds the index of the last appearance of compareString in this StringParser.
* for "abcdef".lastIndexOf("d") this function will return 3
* Returns -1 if compareString isnt even found
*/
int StringParser::lastIndexOf(string compareString)
{
	int lastIndex = -1;
	int size = (int)stringData.size();
	int compareStringSize = (int)compareString.size();

	for (int i = (size - 1); i > 0; i--)
	{
		//get current string in original to compare
		StringParser currentSlice = StringParser(stringData.substr(i - compareStringSize + 1, compareStringSize));
		//compare
		if (currentSlice == compareString) {
			//last index found
			lastIndex = i - compareStringSize + 1;
			break;
		}
	}
	return lastIndex;
}


/**
* For a file path stored in this object, i.e. "C://folder1//folder2//file.txt" this function
* erases the filename to leave the path to the folder only -> "C://folder1//folder2//"
*/
StringParser StringParser::eraseFileNameFromFilePath()
{
	StringParser result("");
	int size = (int) stringData.size();

	for (int i = (size - 1); i > 0; i--)
	{
		StringParser currentSlice = StringParser(stringData.substr(i, 1));
		if (currentSlice == "\\") {
			//last backspace found, return substring
			result = StringParser(stringData.substr(0, i + 1));
			break;
		}
	}

	return result;
}


/**
* For a file path stored in this object, i.e. "C://folder1//folder2//file.txt" this function
* extracts the filename -> "file.txt"
*/
StringParser StringParser::extractFileNameFromFilePath()
{
	StringParser result("");
	int size = (int)stringData.size();

	for (int i = (size - 1); i > -1; i--)
	{
		StringParser currentSlice = StringParser(stringData.substr(i, 1));
		if (currentSlice == "\\" || currentSlice == "/") {
			//last backspace found, return substring
			result = StringParser(stringData.substr(i+1, size-1));
			break;
		}

		//if end is reached, the whole string is valid
		if (i == 0) {
			result = StringParser(stringData);
		}
	}

	return result;
}

/**
* This function returns the argument of a string of the format: "param argument"
* i.e. for a line in a file: "para abc xyz" retrieveArgumentForParam("para") would return "abc xyz"
*/
string StringParser::retrieveArgumentForParam(string paramName)
{
	int paramSize = (int)paramName.size();
	if (paramSize > 0) {
		paramSize += 1; //skip the whitespace
	}
	
	int argSize = (int)stringData.size() - paramSize;

	//ignore ALL whitespaces following the paramName
	int i = paramSize;
	char currentChar = stringData.at(i);
	while (currentChar == ' ') {
		i++;
		currentChar = stringData.at(i);		
	}

	string argument = stringData.substr(i, argSize);
	return argument;
}

/**
* This function returns the arguments of a string of the format: "param argument"
* i.e. for a line in a file: "Ns 0.2" getFloatParam("Ns") would return 0.2f
*/
float StringParser::getFloatParam(string paramName)
{
	string argument = retrieveArgumentForParam(paramName);
	float value = (float) atof(argument.data());
	return value;
}

/**
* This function returns the arguments of a string of the format: "param argument"
* i.e. for a line in a file: "Ns 0.2" getDoubleParam("Ns") would return 0.2
*/
double StringParser::getDoubleParam(string paramName)
{
	string argument = retrieveArgumentForParam(paramName);
	double value = atof(argument.data());
	return value;
}

/**
* This function returns the arguments of a string of the format: "param argument"
* i.e. for a line in a file: "illumination 3" getIntParam("illumination") would return 3
*/
int StringParser::getIntParam(string paramName)
{
	string argument = retrieveArgumentForParam(paramName);
	int value = atoi(argument.data());
	return value;
}

/**
* This function returns the arguments of a string of the format: "param argument"
* i.e. for a line in a file: "Kd 0.2 0.2 0.3" getVec3Param("Kd") would return vec3(0.2, 0.2, 0.3)
*/
vec3 StringParser::getVec3Param(string paramName)
{
	//contains string of floats seperated by whitespaces
	string argument = retrieveArgumentForParam(paramName);
	stringstream f1, f2, f3;
	string f1_s, f2_s, f3_s;
	int i = 0;
	char currentChar = argument.at(i);
	while (currentChar != ' ') {
		currentChar = argument.at(i);
		f1 << currentChar;
		i++;
	}
	f1_s = f1.str();
	//i++;
	currentChar = argument.at(i);
	while (currentChar != ' ') {
		currentChar = argument.at(i);
		f2 << currentChar;
		i++;
	}
	f2_s = f2.str();
	//i++;
	currentChar = argument.at(i);
	while (i < argument.size()) {
		currentChar = argument.at(i);
		f3 << currentChar;
		i++;
	}
	f3_s = f3.str();

	float fl1, fl2, fl3;
	fl1 = (float) atof(f1_s.data());
	fl2 = (float) atof(f2_s.data());
	fl3 = (float)atof(f3_s.data());

	return vec3(fl1, fl2, fl3);
}

/**
* This function returns the arguments of a string of the format: "param argument"
* i.e. for a line in a file: "face 1 2 3" getVec3Param("face") would return ivec3(1, 2, 3)
*/
ivec3 StringParser::getIVec3Param(string paramName)
{
	//contains string of floats seperated by whitespaces
	string argument = retrieveArgumentForParam(paramName);
	stringstream f1, f2, f3;
	string f1_s, f2_s, f3_s;
	int i = 0;
	char currentChar = argument.at(i);
	while (currentChar != ' ') {
		currentChar = argument.at(i);
		f1 << currentChar;
		i++;
	}
	f1_s = f1.str();
	//i++;
	currentChar = argument.at(i);
	while (currentChar != ' ') {
		currentChar = argument.at(i);
		f2 << currentChar;
		i++;
	}
	f2_s = f2.str();
	//i++;
	currentChar = argument.at(i);
	while (i < argument.size()) {
		currentChar = argument.at(i);
		f3 << currentChar;
		i++;
	}
	f3_s = f3.str();

	int i1, i2, i3;
	i1 = atoi(f1_s.data());
	i2 = atoi(f2_s.data());
	i3 = atoi(f3_s.data());

	return ivec3(i1, i2, i3);
}

vec2 StringParser::getVec2Param(string paramName)
{
	//contains string of floats seperated by whitespaces
	string argument = retrieveArgumentForParam(paramName);
	stringstream f1, f2;
	string f1_s, f2_s;
	int i = 0;
	char currentChar = argument.at(i);
	while (currentChar != ' ') {
		currentChar = argument.at(i);
		f1 << currentChar;
		i++;
	}
	f1_s = f1.str();
	//i++;
	currentChar = argument.at(i);
	while (i < argument.size()) {
		currentChar = argument.at(i);
		//check if texture coordinates are given as vec3, then the last component should be ignored
		if (currentChar == ' ') {
			break;
		}
		f2 << currentChar;
		i++;
	}
	f2_s = f2.str();

	float fl1, fl2;
	fl1 = (float) atof(f1_s.data());
	fl2 = (float) atof(f2_s.data());

	return vec2(fl1, fl2);
}

vec2 StringParser::getTexCoordParam(string paramName)
{
	vec2 result = getVec2Param(paramName);
	//now clamp possible texCoords to [0,1]
	if (result.x > 1.0f || result.x < 0.0f) {
		result.x = result.x - (floorf(result.x));
		if (result.x < 0.0f) {
			result.x = 1.0f + result.x;
		}
	}
	if (result.y > 1.0f || result.y < 0.0f) {
		result.y = result.y - (floorf(result.y));
		if (result.y < 0.0f) {
			result.y = 1.0f + result.y;
		}
	}
	return result;
}

void StringParser::split(char seperator, vector<StringParser>* resultStrings)
{
	int i = 0;
	char currentChar;
	stringstream splitString;
	while (i < stringData.size()) {
		currentChar = stringData.at(i);
		if (currentChar == seperator) {
			//store everything read into a string and save it to vector
			string currentString = splitString.str();
			resultStrings->push_back(StringParser(currentString));
			//empty stringstream so it can be filled with new chars
			splitString.clear();
			splitString.str("");
		}
		else {
			splitString << currentChar;
		}		
		i++;
	}
	//store last data into string too
	string currentString = splitString.str();
	if (currentString != "") {
		resultStrings->push_back(StringParser(currentString));
	}
}

void StringParser::append(StringParser newEnd)
{
	stringData.append(newEnd.getStdStringData());
}