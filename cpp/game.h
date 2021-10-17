#if !defined(_GAME_H_)
#define _GAME_H_

#include <vector>
#include <iostream>
#include <string>
#include <unordered_map>
#include <torch/torch.h>
#include <torch/script.h>
#include "model.h"
using namespace std;

extern const vector<char> CARD_CHAR;
extern const unordered_map<char, uint16_t> CARD_IDX;
// rank的索引数组，没有重复
using IdxList = vector<int>;
// 牌组合，长度为15的vector
// 每个元素表示对应rank的数量，不超过4，大小王不超过1
using CardVector = vector<uint16_t>;
#define TWO_IDX 12
#define BJ_IDX 13
#define RJ_IDX 14
#define SIZE 15
#define PLAYER_CNT 3
#define CARD_CNT 54
extern const vector<string> PLAYER_STR;

vector<CardVector> get_actions(CardVector& cards, CardVector& rival_cards);

inline uint16_t next_player(uint16_t curr) {
    return curr == 2 ? 0 : curr + 1;
}

inline uint16_t prev_player(uint16_t curr) {
    return curr == 0 ? 2 : curr - 1;
}

inline int randint(int high);
inline double randd();
int epsilon_greedy(const at::Tensor &q, double epsilon);

void list2vec(const vector<uint16_t>& cards, int start, int stop, CardVector& vec);
void parse_vec(const CardVector& vec, string& str);
void vec_add(CardVector& out, CardVector& a, CardVector& b);
void vec_minus(CardVector& out, CardVector& a, CardVector& b);

struct Observation {
    // 游戏结束标志
    bool done = false;
    // 当前玩家索引
    uint16_t player = 0;
    // 一局游戏内的炸弹出牌次数
    uint16_t bomb_num = 0;
    // 即时奖励
    float reward = 0.0;
    // 当前玩家的手牌
    CardVector cards;
    // 其他玩家手牌之和
    CardVector other_cards;
    // 三张地主牌
    vector<uint16_t> three_cards;
    // 每个玩家手牌张数
    vector<uint16_t> player_cards_cnt;
    // 当前玩家出牌需要压过的牌
    CardVector last_move;
    // 每个玩家最近出的牌
    vector<CardVector> last_moves;
    // 当前玩家的所有有效出牌
    vector<CardVector> valid_moves;
    // 每个玩家出过的牌
    vector<CardVector> played_cards;
    // 所有玩家出牌历史记录
    vector<CardVector> history;
};

at::Tensor cards2tensor(const CardVector& cards);
at::Tensor one_hot(int index, int num_class);
vector<at::Tensor> get_feature(const Observation& obs, const c10::Device& device);

class Agent {
public:
    Agent() = default;
    Agent(const string& name):name(name) {}
    virtual int act(const Observation& obs) { return 0; }
    string name;
};

class RandomAgent : public Agent {
public:
    RandomAgent() { name = "random"; }
    virtual int act(const Observation& obs);
};

class DeepAgent : public Agent {
public:
    DeepAgent(const vector<string>& model_paths, bool jit, const string& device="cpu", double epsilon=0.0);
    virtual int act(const Observation& obs);
    bool jit;
    double epsilon;
    c10::Device device;
    vector<torch::jit::Module> jit_models;
    vector<LstmModel> models;
};

class HumanAgent : public Agent {
public:
    HumanAgent(const string& name):Agent(name) {}
    virtual int act(const Observation& obs);
};

struct GameData {
    uint16_t winner;
    vector<uint16_t> deck;
    vector<string> players;
    vector<CardVector> history;
    vector<CardVector> init_cards;
};

void parse_game_data(size_t id, const GameData& data, const string& save_path);

class Game {
public:
    Game(const string& objective, vector<shared_ptr<Agent>>& players, uint16_t info=0);
    const Observation& reset(const vector<shared_ptr<Agent>>& players={}, const vector<uint16_t>& deck={});
    const Observation& step(int action_idx=-1);
    float get_reward();
    int get_winner();
    bool is_over();
    uint16_t get_bomb_num();
    GameData game_data();
private:
    void init_deck();
    void deal_card();
    const Observation& get_obs();
    // 游戏结束标志
    bool over = false;
    // 0不显示信息 1出牌信息 2出牌信息+所有玩家的手牌信息
    uint16_t info = 0;
    // 奖励机制
    string objective;
    // 三个玩家，按照 landlord,landlord_down,landlord_up顺序
    vector<shared_ptr<Agent>> players;
    // 一副牌
    vector<uint16_t> deck;
    // 当前玩家的观测数据 包括私有信息和公共信息
    Observation obs;
    // 每个玩家手中的牌
    vector<CardVector> player_cards;
    vector<CardVector> init_cards;
};

#endif // _GAME_H_
