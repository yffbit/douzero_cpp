#include <csignal>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <ctime>
#include <cmath>
#include "data_loop.h"
using std::printf;
namespace fs = std::filesystem;

struct Config {
    string xpid = "douzero";
    int save_interval = 30;
    string objective = "adp";
    vector<string> actor_device = {"cuda:0"};
    string training_device = "cuda:0";
    int num_actors = 5;
    bool load_model = false;
    bool disable_checkpoint = false;
    string savedir = "checkpoints";
    long long total_frames = 100000000000;
    float exp_epsilon = 0.01;
    int batch_size = 32;
    int unroll_length = 100;
    int num_buffers = 50;
    // int num_threads = 4;
    float max_grad_norm = 40.;
    float learning_rate = 0.0001;
    float alpha = 0.99;
    float momentum = 0;
    float epsilon = 1e-5;
};

void parse_device(const vector<string>& vec, int offset, vector<string>& device) {
    device.clear();
    int n = vec.size();
    for(; offset < n; offset++) {
        if(vec[offset] == "cpu") device.push_back("cpu");
        else device.push_back("cuda:"+vec[offset]);
    }
}

void parse_config(const char* cfg_path, Config& cfg) {
    ifstream file(cfg_path, ios::in);
    if(!file) {
        printf("%s doesn't exist\n", cfg_path);
        exit(-1);
    }
    else {
        string line;
        while(getline(file, line)) {
            if(line.empty() || line[0] == '#' || line[0] == ' ') continue;
            vector<string> vec = split(line, " =,");
            if(vec.size() < 2) continue;
            string& key = vec[0], & val = vec[1];
            if(key == "xpid") cfg.xpid = val;
            else if(key == "save_interval") cfg.save_interval = stoi(val);
            else if(key == "objective") cfg.objective = val;
            else if(key == "actor_device") parse_device(vec, 1, cfg.actor_device);
            else if(key == "training_device") cfg.training_device = val == "cpu" ? val : "cuda:"+val;
            else if(key == "num_actors") cfg.num_actors = stoi(val);
            else if(key == "load_model" && val == "true") cfg.load_model = true;
            else if(key == "disable_checkpoint" && val == "true") cfg.disable_checkpoint = true;
            else if(key == "savedir") cfg.savedir = val;
            else if(key == "total_frames") cfg.total_frames = stoll(val);
            else if(key == "exp_epsilon") cfg.exp_epsilon = stof(val);
            else if(key == "batch_size") cfg.batch_size = stoi(val);
            else if(key == "unroll_length") cfg.unroll_length = stoi(val);
            else if(key == "num_buffers") cfg.num_buffers = stoi(val);
            // else if(key == "num_threads") cfg.num_threads = stoi(val);
            else if(key == "max_grad_norm") cfg.max_grad_norm = stof(val);
            else if(key == "learning_rate") cfg.learning_rate = stof(val);
            else if(key == "alpha") cfg.alpha = stof(val);
            else if(key == "momentum") cfg.momentum = stof(val);
            else if(key == "epsilon") cfg.epsilon = stof(val);
        }
    }
    file.close();
    if(cfg.num_buffers < cfg.batch_size) {
        printf("<num_buffers> must be no less than <batch_size>.");
        exit(-1);
    }
    assert(cfg.actor_device.size() && cfg.num_actors > 0 && cfg.total_frames > 0);
}

void print_help(const char *exe) {
    printf("%s [config_path]\n", exe);
    printf("xpid, default=douzero, type=str\n");
    printf("save_interval, default=30, type=int\n");
    printf("objective, default=adp, type=str, choices=[adp, wp, logadp]\n");
    printf("actor_device, default=0, type=str\n");
    printf("training_device, default=0, type=str\n");
    printf("gpu_devices, default=0, type=str\n");
    printf("num_actors, default=5, type=int\n");
    printf("load_model, default=false\n");
    printf("disable_checkpoint, default=false\n");
    printf("savedir, default=checkpoints\n");
    printf("total_frames, default=100000000000, type=int\n");
    printf("exp_epsilon, default=0.01, type=float\n");
    printf("batch_size, default=32, type=int\n");
    printf("unroll_length, default=100, type=int\n");
    printf("num_buffers, default=50, type=int\n");
    printf("num_threads, default=4, type=int\n");
    printf("max_grad_norm, default=40., type=float\n");
    printf("learning_rate, default=0.0001, type=float\n");
    printf("alpha, default=0.99, type=float\n");
    printf("momentum, default=0, type=float\n");
    printf("epsilon, default=1e-5, type=float\n");
    exit(-1);
}

Context ctx;
deque<deque<Buffer>> buffers;// [num_devices,PLAYER_CNT]
atomic_bool stop_sig = false;

void signal_handle(int signal) {
    if(signal == SIGINT || signal == SIGTERM) {
        stop_sig.store(true);
        ctx.stop();
        for (auto& vec : buffers) {
            for (auto& buffer : vec) buffer.stop();
        }
    }
}

template<class T, uint16_t N>
class DataStreamMean {// 数据流中最近N个数的均值
public:
    T add(T val) {
        if (n >= N) out = nums[i];
        else out = 0, n++;
        nums[i] = val;
        sum += (val - out);
        if ((++i) == N) i = 0;
        return sum / n;
    }
private:
    uint16_t i = 0, n = 0;
    T nums[N] = { 0 };
    T sum = 0, out = 0;
};

void update_model(LstmModel& model, uint16_t p, vector<vector<ModelLocker>>& mlockers) {
    StateDict params = model->named_parameters(true);
    StateDict buffers = model->named_buffers(true);
    uint16_t m = mlockers.size(), n = mlockers[0].size();
    for (uint16_t i = 0; i < m; i++) {
        for (uint16_t j = 0; j < n; j++) mlockers[i][j].update(p, params, buffers);
    }
}

class TrainLoop : public ThreadLoop {
public:
    TrainLoop(uint16_t player, atomic_llong& frame, float* stat, DataStreamMean<float, 100>& mean_episode_return, mutex& lock, Config& cfg,
        LstmModel& model, torch::optim::RMSprop& optim, Buffer& buffer, vector<vector<ModelLocker>>& mlockers) :
        player(player), frame(frame), stat(stat), mean_episode_return(mean_episode_return), lock(lock), cfg(cfg),
        model(model), optim(optim), device(c10::Device(cfg.training_device)), buffer(buffer), mlockers(mlockers) {}
    virtual void loop() {
        //uint16_t n = mlockers.size();
        while (run && frame < cfg.total_frames) {
            vector<vector<at::Tensor>> batchs(FIELDS);
            vector<at::Tensor>& done = batchs[0];
            while (run && done.size() < cfg.batch_size) {
                vector<at::Tensor> temp = buffer.pop();
                if (temp.size() == FIELDS) {
                    for (uint16_t i = 0; i < FIELDS; i++) batchs[i].push_back(temp[i]);
                }
            }
            if (!run) break;
            vector<at::Tensor> cat_batch;
            for (uint16_t i = 0; i < FIELDS; i++)
                cat_batch.push_back(torch::cat(batchs[i]).to(device));
            at::Tensor x = torch::cat({ cat_batch[3],cat_batch[4] }, 1).to(torch::kF32);
            at::Tensor z = cat_batch[5].to(torch::kF32);
            float episode_return = cat_batch[1].index({ cat_batch[0] }).mean().item<float>();
            try {
                lock_guard<mutex> lk(lock);
                at::Tensor out = model->forward(z, x);
                at::Tensor loss = (out.flatten() - cat_batch[2]).square().mean();
                stat[0] = loss.item<float>();
                if (!isnan(episode_return)) {
                    stat[1] = mean_episode_return.add(episode_return);
                }
                optim.zero_grad();
                loss.backward();
                torch::nn::utils::clip_grad_norm_(model->parameters(), cfg.max_grad_norm);
                optim.step();
                //for (uint16_t i = 0; i < n; i++) mlockers[i]->update(model);
                update_model(model, player, mlockers);
                frame.fetch_add(x.size(0));
            } catch (exception& e) {
                cout << e.what() << endl;
            }
        }
    }
private:
    uint16_t player;
    atomic_llong& frame;
    float* stat;
    mutex& lock;
    Config& cfg;
    LstmModel& model;
    c10::Device device;
    torch::optim::RMSprop& optim;
    Buffer& buffer;
    vector<vector<ModelLocker>>& mlockers;
    DataStreamMean<float, 100>& mean_episode_return;
};

void write_log(ofstream& log, double t0, double t1, atomic_llong frames[PLAYER_CNT], at::Tensor& frame0, at::Tensor& frame1, float stats[PLAYER_CNT][2]) {
    static char str[300] = "";
    at::Tensor avg = (frame1 - frame0) / (t1 - t0);
    long long f0 = frames[0], f1 = frames[1], f2 = frames[2];
    float* avg_f = avg.data_ptr<float>();
    time_t now = time(0);
    tm* local = localtime(&now);
    int n = sprintf(str, "[%d/%02d/%02d %02d:%02d:%02d]", local->tm_year+1900, local->tm_mon+1, local->tm_mday, local->tm_hour, local->tm_min, local->tm_sec);
    n += sprintf(str+n, "loss: L:%.6f D:%.6f U:%.6f mean_episode_return: L:%.6f D:%.6f U:%.6f frame: L:%lld D:%lld U:%lld avg: L:%.2f D:%.2f U:%.2f\n",
        stats[0][0], stats[1][0], stats[2][0], stats[0][1], stats[1][1], stats[2][1], f0, f1, f2, avg_f[0], avg_f[1], avg_f[2]);
    log.write(str, n);
    log.flush();
    cout << str;
}

void checkpoint(string& dir, vector<LstmModel>& models, vector<torch::optim::RMSprop>& optims, mutex locks[PLAYER_CNT], at::Tensor& stats, atomic_llong frames[PLAYER_CNT], at::Tensor& frames_tensor) {
    torch::serialize::OutputArchive archive;
    for (uint16_t p = 0; p < PLAYER_CNT; p++) locks[p].lock();
    for (uint16_t p = 0; p < PLAYER_CNT; p++) {
        auto params = models[p]->named_parameters(true);
        auto buffers = models[p]->named_buffers(true);
        string prefix = to_string(p) + '_';
        for (auto& val : params) archive.write(prefix + val.key(), val.value());
        for (auto& val : buffers) archive.write(prefix + val.key(), val.value(), true);
        torch::save(models[p], dir + "cppmodel_" + prefix + to_string(frames[p]) + ".pt");
        torch::save(optims[p], dir + prefix + "optim.tar");
    }
    archive.write("stats", stats, true);
    archive.write("frames", frames_tensor, true);
    for (uint16_t p = 0; p < PLAYER_CNT; p++) locks[p].unlock();
    string checkpoint_path = dir + "model.tar";
    archive.save_to(checkpoint_path);
    cout << "Saving checkpoint to " << checkpoint_path << endl;
}

bool keep_run(atomic_llong frames[PLAYER_CNT], long long total_frames) {
    for (uint16_t p = 0; p < PLAYER_CNT; p++)
        if (frames[p] < total_frames) return true;
    return false;
}

int main(int argc, const char* argv[]) {
    if(argc > 2) print_help(argv[0]);
    try {
        std::signal(SIGINT, signal_handle);
        std::signal(SIGTERM, signal_handle);
        Config cfg;
        if(argc == 2) parse_config(argv[1], cfg);
        torch::manual_seed(time(0));
        // 训练线程模型
        vector<LstmModel> train_model;
        torch::optim::RMSpropOptions rms_options(cfg.learning_rate);
        rms_options.alpha(cfg.alpha);
        rms_options.momentum(cfg.momentum);
        rms_options.eps(cfg.epsilon);
        vector<torch::optim::RMSprop> optims;
        mutex locks[PLAYER_CNT];
        c10::Device training_device(c10::Device(cfg.training_device));
        int lstm_input = 162, lstm_hidden = 128;
        int dim[PLAYER_CNT] = {373, 484, 484};
        for (uint16_t p = 0; p < PLAYER_CNT; p++) {
            train_model.emplace_back(lstm_input, lstm_hidden, dim[p] + lstm_hidden);
            train_model[p]->to(training_device);
            // train_model[p]->train();
            optims.emplace_back(train_model[p]->parameters(), rms_options);
        }
        // 记录训练状态
        atomic_llong frames[PLAYER_CNT] = { 0,0,0 };
        float stats[PLAYER_CNT][2] = { {0.0,0.0},{0.0,0.0},{0.0,0.0} };// loss,mean_episode_return
        string checkpoint_dir = cfg.savedir + '/' + cfg.xpid + '/';
        if (!fs::exists(checkpoint_dir)) fs::create_directories(checkpoint_dir);
        string checkpoint_path = checkpoint_dir + "model.tar";
        at::Tensor stats_tensor = torch::from_blob(stats, { PLAYER_CNT,2 }, torch::kF32);
        at::Tensor frames_tensor = torch::from_blob(frames, { PLAYER_CNT }, torch::kLong);
        vector<DataStreamMean<float, 100>> mean_episode_returns(PLAYER_CNT);
        if (cfg.load_model && fs::exists(checkpoint_path)) {// 载入训练状态
            torch::serialize::InputArchive archive;
            archive.load_from(checkpoint_path, training_device);
            for (uint16_t p = 0; p < PLAYER_CNT; p++) {
                auto params = train_model[p]->named_parameters(true);
                auto buffers = train_model[p]->named_buffers(true);
                string prefix = to_string(p) + '_';
                for (auto& val : params) archive.read(prefix + val.key(), val.value());
                for (auto& val : buffers) archive.read(prefix + val.key(), val.value(), true);
                torch::serialize::InputArchive optim_archive;
                optim_archive.load_from(checkpoint_dir + prefix + "optim.tar", training_device);
                optims[p].load(optim_archive);
            }
            at::Tensor temp1, temp2;
            archive.read("stats", temp1, true);
            stats_tensor.copy_(temp1);
            cout << stats_tensor << endl;
            archive.read("frames", temp2, true);
            frames_tensor.copy_(temp2);
            cout << frames_tensor << endl;
            for (uint16_t p = 0; p < PLAYER_CNT; p++) mean_episode_returns[p].add(stats[p][1]);
            cout << "Load checkpoint " << checkpoint_path << endl;
        }
        for (uint16_t p = 0; p < PLAYER_CNT; p++) train_model[p]->train();
        // 模拟线程模型
        int n = cfg.actor_device.size();// gpu/cpu设备数量
        buffers.resize(n);
        vector<vector<vector<LstmModel>>> models(n, vector<vector<LstmModel>>(cfg.num_actors));
        vector<vector<ModelLocker>> mlockers(n);
        for (uint16_t i = 0; i < n; i++) {
            c10::Device device = c10::Device(cfg.actor_device[i]);
            for (uint16_t p = 0; p < PLAYER_CNT; p++)
                buffers[i].emplace_back(cfg.num_buffers);
            for (uint16_t j = 0; j < cfg.num_actors; j++) {
                vector<LstmModel>& models_i_j = models[i][j];
                for (uint16_t p = 0; p < PLAYER_CNT; p++) {
                    models_i_j.emplace_back(lstm_input, lstm_hidden, dim[p] + lstm_hidden);
                    models_i_j[p]->to(device);
                    models_i_j[p]->train(false);
                }
                mlockers[i].emplace_back(models_i_j);
            }
        }
        // 复制模型参数
        for (uint16_t p = 0; p < PLAYER_CNT; p++) update_model(train_model[p], p, mlockers);
        // 创建线程
        uint16_t player = 0, T = cfg.unroll_length;
        double epsilon = cfg.exp_epsilon;
        string& objective = cfg.objective;
        for (uint16_t i = 0; i < n; i++) {
            for (uint16_t j = 0; j < cfg.num_actors; j++) {
                shared_ptr<ThreadLoop> loop = make_shared<DataLoop>(player, T, epsilon, objective, cfg.actor_device[i],
                                                                    mlockers[i][j], buffers[i]);
                ctx.push(loop);
                player = next_player(player);
            }
            for (uint16_t p = 0; p < PLAYER_CNT; p++) {
                shared_ptr<ThreadLoop> loop = make_shared<TrainLoop>(p, frames[p], stats[p], mean_episode_returns[p], locks[p], cfg, train_model[p],
                                                                        optims[p], buffers[i][p], mlockers);
                ctx.push(loop);
            }
        }
        Timer timer;
        ofstream log_file(checkpoint_dir + "train_log.txt", ios::app);
        log_file << "time loss mean_episode_return frame avg_speed\n";
        
        double t0 = timer.duration(), t1, last_save, interval = cfg.save_interval * 60.0;
        at::Tensor frame0 = frames_tensor.clone();
        last_save = t0;
        ctx.start();
        long long total_frames = cfg.total_frames;
        while (!stop_sig && keep_run(frames, total_frames)) {
            this_thread::sleep_for(5s);
            t1 = timer.duration();
            write_log(log_file, t0, t1, frames, frame0, frames_tensor, stats);
            if (t1 - last_save >= interval) {
                if(!cfg.disable_checkpoint)
                    checkpoint(checkpoint_dir, train_model, optims, locks, stats_tensor, frames, frames_tensor);
                last_save = t1;
            }
        }
        ctx.join();
        write_log(log_file, t0, timer.duration(), frames, frame0, frames_tensor, stats);
        log_file << endl;
        log_file.close();
        if(!cfg.disable_checkpoint)
            checkpoint(checkpoint_dir, train_model, optims, locks, stats_tensor, frames, frames_tensor);
        exit(0);
    } catch(exception& e) {
        printf("%s\n", e.what());
        exit(-1);
    }
}