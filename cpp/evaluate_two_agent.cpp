#include <fstream>
#include <thread>
#include <mutex>
#include <atomic>
#include <ctime>
#include <csignal>
#include <deque>
#include <filesystem>
#include <ctime>
#include "util.h"
#include "game.h"
#include "model_locker.h"
using namespace std;
namespace fs = std::filesystem;

struct Config {
    Config() :agent1_path(PLAYER_CNT), agent2_path(PLAYER_CNT) {}
    // string objective = "wp";
    string device = "cuda:0";
    bool agent1_jit = true;
    bool agent2_jit = true;
    vector<string> agent1_path;
    vector<string> agent2_path;
    uint32_t num_threads = 4;
    uint32_t num_games = 5000;
    string save_path;
};

char str[100] = "";
Queue<GameData> data_q;
Config cfg;
mutex mlock;
uint32_t total_wins[2] = { 0,0 };
long long total_scores[2] = { 0,0 };
atomic_uint num_games = 0;
atomic_bool stop_sig = false;

void signal_handle(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        stop_sig.store(true);
    }
}

void print_help(const char* exe) {
    cout << exe << " config_path\n";
    // cout << "objective=wp\n";
    cout << "device=cuda:0\n";
    cout << "agent1_0=agent1 landlord model path\n";
    cout << "agent1_1=agent1 landlord_down model path\n";
    cout << "agent1_2=agent1 landlord_up model path\n";
    cout << "agent1_jit=true\n";
    cout << "agent2_0=agent2 landlord model path\n";
    cout << "agent2_1=agent2 landlord_down model path\n";
    cout << "agent2_2=agent2 landlord_up model path\n";
    cout << "agent2_jit=true\n";
    cout << "num_threads=4\n";
    cout << "num_games=5000\n";
    cout << "save_path=game_data/test.txt\n";
    exit(-1);
}

void parse_config(const char* cfg_path) {
    ifstream file(cfg_path, ios::in);
    if (!file) {
        cout << cfg_path << " doesn't exist.\n";
        exit(-1);
    }
    string line;
    while (getline(file, line)) {
        if (line.empty() || line[0] == '#' || line[0] == ' ') continue;
        vector<string> vec = split(line, '=');
        if (vec.size() < 2) continue;
        string& key = vec[0], & val = vec[1];
        if (key == "device") cfg.device = val;
        else if(key == "agent1_0") cfg.agent1_path[0] = val;
        else if(key == "agent1_1") cfg.agent1_path[1] = val;
        else if(key == "agent1_2") cfg.agent1_path[2] = val;
        else if(key == "agent1_jit" && val == "false") cfg.agent1_jit = false;
        else if(key == "agent2_0") cfg.agent2_path[0] = val;
        else if(key == "agent2_1") cfg.agent2_path[1] = val;
        else if(key == "agent2_2") cfg.agent2_path[2] = val;
        else if(key == "agent2_jit" && val == "false") cfg.agent2_jit = false;
        else if(key == "num_threads") cfg.num_threads = stoi(val);
        else if(key == "num_games") cfg.num_games = stoi(val);
        else if(key == "save_path") cfg.save_path = val;
    }
    file.close();
}

void write_str(ofstream& file, const char* prefix, uint32_t wins[2], long long scores[2]) {
    file << prefix;
    uint32_t sum = wins[0] + wins[1];
    int n = sprintf(str, ":[%d,%d]/%d,[%lld,%lld]\n", wins[0], wins[1], sum, scores[0], scores[1]);
    file.write(str, n);
    cout << prefix << str;
}

bool check_model_path(const vector<string>& model_path) {
    for (uint16_t p = 0; p < PLAYER_CNT; p++)
        if (!fs::exists(model_path[p])) return false;
    return true;
}

void game_loop(uint16_t id, ofstream& file, bool save=true) {
    shared_ptr<Agent> agent1 = make_shared<RandomAgent>(), agent2 = make_shared<RandomAgent>();
    if (check_model_path(cfg.agent1_path))
        agent1 = make_shared<DeepAgent>(cfg.agent1_path, cfg.agent1_jit, cfg.device);
    if (check_model_path(cfg.agent2_path))
        agent2 = make_shared<DeepAgent>(cfg.agent2_path, cfg.agent2_jit, cfg.device);
    // 两个agent轮流当地主
    vector<vector<shared_ptr<Agent>>> players = { {agent1,agent2,agent2},{agent2,agent1,agent1} };
    Game game("wp", players[0]);
    uint32_t wins[2] = { 0,0 };
    long long scores[2] = { 0,0 };
    GameData two_games[2];
    while (!stop_sig && num_games > 0) {
        uint32_t temp_wins[2] = { 0,0 };
        long long temp_scores[2] = { 0,0 };
        // 同样的牌,双方会交换身份,一共进行两次游戏
        for (uint16_t turn = 0; turn < 2; turn++) {// agent1的身份:0地主,1农民
            if (turn == 0) game.reset(players[turn]);
            else game.reset(players[turn], two_games[0].deck);
            while (!game.is_over()) game.step();
            uint16_t winner = game.get_winner();
            if (winner != 0) winner = 1;// 农民胜
            int score = 1 << (1 + game.get_bomb_num());
            uint16_t idx = turn == winner ? 0 : 1;
            temp_wins[idx]++;
            temp_scores[idx] += score;
            temp_scores[1 - idx] -= score;
            two_games[turn] = game.game_data();
        }
        {
            lock_guard<mutex> lk(mlock);
            if (num_games == 0) break;
            num_games--;
            for (uint16_t i = 0; i < 2; i++) {
                wins[i] += temp_wins[i];
                scores[i] += temp_scores[i];
            }
            if (save) {
                data_q.push(two_games[0]);
                data_q.push(two_games[1]);
            }
        }
    }
    string s = "thread " + to_string(id);
    mlock.lock();
    write_str(file, s.c_str(), wins, scores);
    for (uint16_t i = 0; i < 2; i++) {
        total_wins[i] += wins[i];
        total_scores[i] += scores[i];
    }
    mlock.unlock();
}

void save_loop() {
    size_t cnt = 0;
    while (true) {
        try {
            GameData temp = data_q.pop();
            parse_game_data(++cnt, temp, cfg.save_path);
        } catch(exception& e) {
            // cout << e.what() << endl;
            break;
        }
    }
}

int main(int argc, const char* argv[]) {
    if (argc != 2) print_help(argv[0]);
    parse_config(argv[1]);
    assert(cfg.num_games > 0 && cfg.num_threads > 0);
    signal(SIGINT, signal_handle);
    signal(SIGTERM, signal_handle);
    num_games.store(cfg.num_games);
    ofstream file(argv[1], ios::app);
    file << "\n\n" << "result:\n";
    bool save = false;
    if (!cfg.save_path.empty()) {
        save = true;
        fs::path temp(cfg.save_path);
        if (fs::is_directory(temp) || !temp.has_filename())
            temp /= to_string(time(0)) + ".txt";
        fs::create_directories(temp.parent_path());
        cfg.save_path = temp.string();
    }
    Timer timer;
    vector<thread> threads;
    for (uint16_t i = 0; i < cfg.num_threads; i++)
        threads.emplace_back(game_loop, i, std::ref(file), save);
    if (save) threads.emplace_back(save_loop);
    for (uint16_t i = 0; i < cfg.num_threads; i++)
        threads[i].join();
    data_q.set();
    if (save) threads.back().join();
    write_str(file, "total", total_wins, total_scores);
    double total = total_wins[0] + total_wins[1];
    double avg[4] = { 0,0,0,0 };
    if (total != 0) {
        avg[0] = total_wins[0] / total;
        avg[1] = total_wins[1] / total;
        avg[2] = total_scores[0] / total;
        avg[3] = total_scores[1] / total;
    }
    int n = sprintf(str, "avg:[%f,%f],[%f,%f]\ntime:%.2f\n", avg[0], avg[1], avg[2], avg[3], timer.duration());
    file.write(str, n);
    file.close();
    cout << str;
    return 0;
}