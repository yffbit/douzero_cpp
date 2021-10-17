#if !defined(_UTIL_H_)
#define _UTIL_H_

#include <vector>
#include <deque>
#include <mutex>
#include <atomic>
#include <string>
#include <chrono>
using namespace std;

class Timer {
public:
    Timer() { start(); }
    void start() {
        t0 = chrono::steady_clock::now();
    }
    double duration() {
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        chrono::duration<double> span(t1 - t0);
        return span.count();
    }
private:
    chrono::steady_clock::time_point t0;
};

vector<string> split(const string& str, const string& delimiter) {
    vector<string> ans;
    size_t i = str.find_first_not_of(delimiter), j;
    while(i != string::npos) {
        j = str.find_first_of(delimiter, i+1);
        ans.push_back(str.substr(i, j-i));
        i = str.find_first_not_of(delimiter, j);
    }
    return ans;
}

vector<string> split(const string& str, const char delimiter) {
    vector<string> ans;
    size_t i = str.find_first_not_of(delimiter), j;
    while(i != string::npos) {
        j = str.find_first_of(delimiter, i+1);
        ans.push_back(str.substr(i, j-i));
        i = str.find_first_not_of(delimiter, j);
    }
    return ans;
}

template<class T>
class Queue {
public:
    Queue() = default;
    void push(T val) {
        // if(stop) return;
        {
            lock_guard<mutex> lk(m);
            dq.push_back(val);
        }
        cv.notify_one();
    }
    T pop(bool wait = true) {
        unique_lock<mutex> lk(m);
        if (wait) cv.wait(lk, [this]() { return stop || !dq.empty(); });
        if (dq.empty()) throw std::out_of_range("container is empty");
        T val = dq.front();
        dq.pop_front();
        return val;
    }
    void set(bool stop = true) {
        this->stop = stop;
        cv.notify_all();
    }
private:
    atomic_bool stop = false;
    mutex m;
    condition_variable cv;
    deque<T> dq;
};

#endif // _UTIL_H_
