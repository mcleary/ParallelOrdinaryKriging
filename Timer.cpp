
#include "Timer.h"

using namespace std::chrono;

Timer::Timer()
{
    start();
}

void Timer::start()
{
    m_StartTime = steady_clock::now();
    m_bRunning = true;
}

void Timer::stop()
{
    m_EndTime = steady_clock::now();
    m_bRunning = false;
}

double Timer::elapsedMilliseconds()
{
    return duration_cast<milliseconds>(_elapsed()).count();
}

double Timer::elapsedSeconds()
{
    return duration_cast<seconds>(_elapsed()).count();
}

steady_clock::duration Timer::_elapsed() const
{
    decltype(m_StartTime) endTime;

    if(m_bRunning)
    {
        endTime = steady_clock::now();
    }
    else
    {
        endTime = m_EndTime;
    }

    return endTime - m_StartTime;
}
