#include <iostream>
#include <thread>
#include "Police.h"
#include "Criminal.h"

// The police initiates the exchange, and condition variables put each thread to sleep until its predicate is true.
// This ordered handshake removes the circular busy-wait while mutexes protect the shared state.
int main()
{
    std::cout<<"Main thread start !\n";
    Criminal criminal;
    Police police;
    std::jthread thread_1(&Police::giveRansom, &police, std::ref(criminal));
    std::jthread thread_2(&Criminal::releaseHostage, &criminal, std::ref(police));
    std::cout<<"Main thread end !\n";
}
