#if !defined(_DADA_LOOP_H_)
#define _DADA_LOOP_H_

#include <thread>
#include <queue>
#include "util.h"
#include "game.h"
#include "model_locker.h"

#define FIELDS 6
// 0 done
// 1 episode_return
// 2 target
// 3 obs_x_no_action
// 4 obs_action
// 5 obs_z

class Buffer {
public:
    Buffer(uint16_t num):data(num, vector<at::Tensor>(FIELDS, torch::zeros({}, torch::kI8))) {
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
        full_idxs.push(idx);
    }
    vector<at::Tensor> pop(bool wait = true) {// 消费者调用
        try {
            uint16_t idx = full_idxs.pop(wait);
            vector<at::Tensor> temp = data[idx];
            free_idxs.push(idx);
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

class ThreadLoop {
public:
    ThreadLoop() = default;
    virtual void loop() {}
    void stop() { run = false; }
protected:
    atomic_bool run = true;
};

class DataLoop : public ThreadLoop {
public:
    DataLoop(uint16_t player, uint16_t T, double epsilon, const string& objective, const string& device,
        ModelLocker& mlocker, deque<Buffer>& buffers):
        player(player), T(T), epsilon(epsilon), objective(objective), mlocker(mlocker),
        buffers(buffers), device(c10::Device(device)) {
            assert(buffers.size() == PLAYER_CNT);
            at::TensorOptions opt = at::TensorOptions(device);
            options.emplace_back(move(opt.dtype(torch::kBool)));
            options.emplace_back(move(opt.dtype(torch::kF32)));
            options.emplace_back(move(opt.dtype(torch::kF32)));
        }
    virtual void loop() {
        vector<vector<queue<at::Tensor>>> data(PLAYER_CNT, vector<queue<at::Tensor>>(FIELDS));
        vector<shared_ptr<Agent>> players(PLAYER_CNT, NULL);
        Game game(objective, players);
        short p = 0, idx = 0, diff;
        while(run) {
            const Observation& obs = game.reset();
            while(!game.is_over()) {
                p = obs.player;
                vector<queue<at::Tensor>>& data_p = data[p];
                vector<at::Tensor> feature = get_feature(obs, device);
                data_p[3].push(feature[0]);
                data_p[5].push(feature[1]);
                idx = 0;
                if(obs.valid_moves.size() > 1) {
                    at::Tensor q = mlocker.forward(p, feature[3], feature[2]);
                    idx = epsilon_greedy(q, epsilon);
                }
                const CardVector& action = obs.valid_moves[idx];
                data_p[4].push(cards2tensor(action).to(device));
                game.step(idx);
            }
            // std::printf("%d,game over,%lld\n", __LINE__, obs.history.size());
            // 游戏结束,设置样本的label
            for(p = 0; p < PLAYER_CNT; p++) {
                vector<queue<at::Tensor>>& data_p = data[p];
                diff = data_p[3].size() - data_p[2].size();
                if(diff == 0) continue;
                float reward = p == 0 ? obs.reward : -obs.reward;
                while((--diff) > 0) {
                    data_p[0].push(torch::tensor(0, options[0]));
                    data_p[1].push(torch::tensor(0.0, options[1]));
                    data_p[2].push(torch::tensor(reward, options[2]));
                }
                data_p[0].push(torch::tensor(1, options[0]));
                data_p[1].push(torch::tensor(reward, options[1]));
                data_p[2].push(torch::tensor(reward, options[2]));
            }
            // p = player;
            // do {
            //     vector<queue<at::Tensor>>& data_p = data[p];
            //     queue<at::Tensor>& done_p = data_p[0];
            //     while(done_p.size() >= T) {
            //         vector<at::Tensor> vec;
            //         for(uint16_t i = 0; i < FIELDS; i++) {
            //             vector<at::Tensor> temp;
            //             queue<at::Tensor>& data_p_q = data_p[i];
            //             for(uint16_t j = 0; j < T; j++) {
            //                 temp.push_back(data_p_q.front());
            //                 data_p_q.pop();
            //             }
            //             vec.push_back(torch::stack(temp));
            //         }
            //         if(!run) break;
            //         buffers[p].push(vec);
            //     }
            //     p = next_player(p);
            // } while((p != player) && run);
            while (run) {
                uint16_t temp_idx = get_player(data);
                if (temp_idx == PLAYER_CNT) break;
                p = temp_idx;
                do {
                    vector<queue<at::Tensor>>& data_p = data[p];
                    queue<at::Tensor>& done_p = data_p[0];
                    while(done_p.size() >= T && run) {
                        short lock_idx = buffers[p].lock(false);// 不阻塞
                        if (lock_idx == -1) break;
                        vector<at::Tensor> vec;
                        for(uint16_t i = 0; i < FIELDS; i++) {
                            vector<at::Tensor> temp;
                            queue<at::Tensor>& data_p_q = data_p[i];
                            for(uint16_t j = 0; j < T; j++) {
                                temp.push_back(data_p_q.front());
                                data_p_q.pop();
                            }
                            vec.push_back(torch::stack(temp));
                        }
                        buffers[p].unlock(vec, lock_idx);
                    }
                    p = next_player(p);
                } while (p != temp_idx && run);
            }
        }
    }
private:
    inline uint16_t get_player(vector<vector<queue<at::Tensor>>>& data) {
        for (uint16_t p = 0; p < PLAYER_CNT; p++)
            if (data[p][3].size() >= T) return p;
        return PLAYER_CNT;
    }
    uint16_t player;
    uint16_t T;
    double epsilon;
    string objective;
    c10::Device device;
    vector<at::TensorOptions> options;
    ModelLocker& mlocker;
    deque<Buffer>& buffers;
};

class Context {
public:
    Context():started(false) {}
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    ~Context() {
        stop();
        join();
    }
    void push(shared_ptr<ThreadLoop> loop) {
        assert(!started);
        loops.push_back(move(loop));
    }
    void start() {
        int n = loops.size();
        for(int i = 0; i < n; i++) {
            threads.emplace_back([this, i]() {
                loops[i]->loop();
            });
        }
        started = true;
    }
    void stop() {
        int n = loops.size();
        for(int i = 0; i < n; i++) loops[i]->stop();
    }
    void join() {
        int n = threads.size();
        for(int i = 0; i < n; i++) threads[i].join();
        for(int i = 0; i < n; i++) {
            threads.pop_back();
            loops.pop_back();
        }
    }
private:
    bool started;
    vector<shared_ptr<ThreadLoop>> loops;
    vector<thread> threads;
};

#endif // _DADA_LOOP_H_
