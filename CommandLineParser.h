
#pragma once

#include <string>

class CommandLineParser
{
public:
    CommandLineParser(int ArgC, char* ArgV[]);
    
    bool OptionExists(const std::string& Option) const;
    
    std::string GetOptionValue(const std::string& Option) const;
    
private:
    int     m_ArgC;
    char**  m_ArgV;
};