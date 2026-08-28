#pragma once
#include <mutex>
#include <iostream>
#include <string>

class Friend
{
    public:
        Friend(std::string friendName):name{friendName}{}

        void bow(Friend &bower)
        {
            std::scoped_lock dataLock{m_mutex, bower.m_mutex};
            std::cout<<name<<": "<<bower.getName()<<" bowed to me!\n";
            std::cout<<bower.getName()<<": "<<name<<" bowed back to me!\n";
        }
        std::string getName()
        {
            return name;
        }
        void setName(std::string friendName)
        {
            name = friendName;
        }
    private:
        std::mutex m_mutex;
        std::string name;
};
