//
//  XYZFile.cpp
//  ParallelDTM
//
//  Created by Thales Sabino on 8/25/16.
//  Copyright © 2016 Thales Sabino. All rights reserved.
//

#include "XYZFile.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>

using namespace std;

PointVector ReadXYZFile(const std::string& FilePath)
{
    PointVector Data;
    
    ifstream InputFile(FilePath, ios::in);
    
    cout << "Reading file ... " << flush;
    string FileContents(istreambuf_iterator<char>(InputFile), (istreambuf_iterator<char>()));
    stringstream FileContentsStream(FileContents);
    cout << "done. Parsing ... " << flush;
    
    string Line;
    while (getline(FileContentsStream, Line))
    {
        istringstream iss(Line);
        vector<string> Tokens{ istream_iterator<string>{iss}, istream_iterator<string>{} };
        
        if (Tokens.size() == 3 || Tokens.size() == 6)
        {
            Data.emplace_back(stof(Tokens[0]), stof(Tokens[1]), stof(Tokens[2]));
        }
        else if (Tokens.size() == 8)
        {
            Data.emplace_back(stof(Tokens[2]), stof(Tokens[3]), stof(Tokens[4]));
        }
    }
    cout << "done" << endl;
    
    return Data;
}

void WriteXYZFile(const std::string& Filepath, PointVector Points)
{
    cout << "Writing to " << Filepath << " ... " << flush;
    ofstream OutputFile(Filepath);
    for(auto Point : Points)
    {
        OutputFile << Point.x << " " << Point.y << " " << Point.z << endl;
    }
    cout << "done" << endl;
}
