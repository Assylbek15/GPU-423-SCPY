#include <iostream>
#include <thread>
#include <vector>
#include "Starvation.h"

// A condition variable schedules the threads in round-robin order, giving every thread a turn.
// Advancing the shared turn after each increment prevents any thread from monopolizing the mutex.
int main()
{
    std::cout<<"Main thread start !\n";
    Starvation starvation;
    {
        std::vector<std::jthread> threads;
        for (int i{0}; i < Starvation::numberOfThreads; ++i)
        {
            threads.push_back(std::jthread{&Starvation::incrementValue, &starvation, i});
        }
    }
    std::cout<<"Main thread end !\n";
    return 0;
}
