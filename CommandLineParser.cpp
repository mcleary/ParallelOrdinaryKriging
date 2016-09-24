
#include "CommandLineParser.h"

#include <algorithm>

CommandLineParser::CommandLineParser(int ArgC, char* ArgV[]) :
    m_ArgC(ArgC),
    m_ArgV(ArgV)
{
}

bool CommandLineParser::OptionExists(const std::string& Option) const
{
    auto Begin = m_ArgV;
    auto End = m_ArgV + m_ArgC;
    return std::find(Begin, End, Option) != End;
}

std::string CommandLineParser::GetOptionValue(const std::string &Option) const
{
    auto Begin = m_ArgV;
    auto End = m_ArgV + m_ArgC;
    
    char ** Itr = std::find(Begin, End, Option);
    
    if (Itr != End && ++Itr != End)
    {
        return *Itr;
    }
    
    return std::string();
}