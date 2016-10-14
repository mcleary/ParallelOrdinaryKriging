#pragma once

#include <chrono>

class Timer
{
public:
    explicit Timer();

    void start();
    void stop();

    double elapsedMilliseconds();
    double elapsedSeconds();

private:
    std::chrono::steady_clock::duration _elapsed() const;

    std::chrono::steady_clock::time_point m_StartTime;
    std::chrono::steady_clock::time_point m_EndTime;
    bool                                  m_bRunning = false;
};
