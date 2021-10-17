#include <vector>
#include <deque>
#include <mutex>
#include <atomic>
#include <thread>
#include <cassert>
#include <torch/torch.h>
#include <torch/script.h>
using namespace std;

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

#define FIELDS 3

class Buffer {
public:
    Buffer(uint16_t num):data(num, vector<at::Tensor>(FIELDS, torch::zeros({}, torch::kI32))) {
        for(uint16_t i = 0; i < num; i++) free_idxs.push(i);
    }
    bool push(const vector<at::Tensor>& vec, bool wait=true) {// 生产者调用
        assert(vec.size() == FIELDS);
        try {
            uint16_t idx = free_idxs.pop(wait);
            data[idx] = vec;
            full_idxs.push(idx);
            return true;
        } catch(exception& e) {
            // cout << e.what() << endl;
            return false;
        }
    }
    short lock(bool wait = true) {// 生产者调用,和unlock函数配套
        try {
            return free_idxs.pop(wait);
        } catch(exception& e) {
            return -1;
        }
    }
    void unlock(const vector<at::Tensor>& vec, uint16_t idx) {// 生产者调用,没有检查索引的正确性
        assert(vec.size() == FIELDS);
        data[idx] = vec;
        //if (idx == 0) {
        //    at::Tensor& temp = data[idx][3];
        //    cout << temp.nonzero().transpose(0,1) << endl;
        //}
        cout << "push" << endl;
        for (int j = 0; j < FIELDS; j++) cout << data[idx][j] << endl;
        full_idxs.push(idx);
    }
    vector<at::Tensor> pop(bool wait = true) {// 消费者调用
        try {
            uint16_t idx = full_idxs.pop(wait);
            vector<at::Tensor> temp = data[idx];
            free_idxs.push(idx);
            //if (idx == 0)
            //    cout << temp[3].nonzero().transpose(0, 1) << endl;
            return temp;
        } catch(exception& e) {
            // cout << e.what() << endl;
            return {};
        }
    }
    void stop() {
        free_idxs.set();
        full_idxs.set();
    }
private:
    Queue<uint16_t> full_idxs;
    Queue<uint16_t> free_idxs;
    vector<vector<at::Tensor>> data;
};

void loop1(Buffer& buffer) {
    for (int i = 0; i < 100; i += 16) {
        int idx = buffer.lock();
        if (idx != -1) {
            vector<at::Tensor> vec(FIELDS);
            vec[0] = torch::range(i, i + 3, torch::kI32).reshape({ 2,2 });
            vec[1] = torch::range(i + 3, i + 8, torch::kI32).reshape({ 2,3 });
            vec[2] = torch::range(i + 8, i + 15, torch::kI32).reshape({ 2,4 });
            //cout << "loop 1" << endl;
            //for (int j = 0; j < FIELDS; j++) cout << vec[j] << endl;
            buffer.unlock(vec, idx);
        }
    }
}

void loop2(Buffer& buffer, atomic_bool& stop) {
    while (!stop) {
        vector<at::Tensor> vec = buffer.pop();
        if (vec.size() == FIELDS) {
            this_thread::sleep_for(1s);
            cout << "loop 2" << endl;
            for (int j = 0; j < FIELDS; j++) cout << vec[j] << endl;
        }
    }
}

int main() {
    Buffer buffer(1);
    atomic_bool stop = false;
    thread t1 = thread(loop1, std::ref(buffer));
    thread t2 = thread(loop2, std::ref(buffer), std::ref(stop));
    t1.join();
    buffer.stop();
    this_thread::sleep_for(1200ms);
    stop.store(true);
    t2.join();
    cout << "exit" << endl;
    return 0;
}