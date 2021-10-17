#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>
#include <cassert>
#include <fstream>
#include "game.h"

// 'T':10, '<':小王, '>':大王
const vector<char> CARD_CHAR = {'3','4','5','6','7','8','9','T','J','Q','K','A','2','<','>'};
const unordered_map<char, uint16_t> CARD_IDX = {
    {'3',0},
    {'4',1},
    {'5',2},
    {'6',3},
    {'7',4},
    {'8',5},
    {'9',6},
    {'T',7},
    {'J',8},
    {'Q',9},
    {'K',10},
    {'A',11},
    {'2',12},
    {'<',13},
    {'>',14},
    {'t',7},
    {'j',8},
    {'q',9},
    {'k',10},
    {'a',11}
};
const vector<string> PLAYER_STR = {"*0", " 1", " 2"};

// 从不重复的候选列表里面选出k张牌
void combine(IdxList& candidate, int k, uint16_t cnt, CardVector& base, vector<CardVector>& res) {
    int i = 0, n = candidate.size();
    vector<int> idx(k, -1);// 选中的数在数组candidate中的索引
    while(i >= 0) {
        idx[i]++;
        if(idx[i]+k-1-i >= n) --i;// 后面还需要选择k-1-i个数
        else if(i == k-1) {
            CardVector temp = base;
            for(int j = 0; j < k; j++) temp[candidate[idx[j]]] = cnt;
            res.emplace_back(move(temp));
        }
        else {
            idx[i+1] = idx[i];++i;
        }
    }
}

void add_single_rank(int i, uint16_t cnt, vector<CardVector>& res) {// 一种牌
    CardVector temp(SIZE, 0);
    temp[i] = cnt;
    res.emplace_back(move(temp));
}
void add_rocket(CardVector& cards, vector<CardVector>& res) {// 大小王
    if(cards[BJ_IDX] && cards[RJ_IDX]) {
        CardVector temp(SIZE, 0);
        temp[BJ_IDX] = temp[RJ_IDX] = 1;
        res.emplace_back(move(temp));
    }
}
void add_bomb(IdxList& idx4, vector<CardVector>& res) {// 炸弹
    int n = idx4.size();
    for(int i = 0; i < n; i++) add_single_rank(idx4[i], 4, res);
}

// 给指定牌增加翼
void add_single_wing(IdxList& idxs, int idx1, int idx2, uint16_t cnt, CardVector& base, vector<CardVector>& res) {
    int n = idxs.size(), idx;
    for(int i = 0; i < n; i++) {
        idx = idxs[i];
        if(idx >= idx1 && idx <= idx2) continue;// 跳过区间[idx1,idx2]
        CardVector temp = base;
        temp[idx] = cnt;
        res.emplace_back(move(temp));
    }
}
void add_multi_wing(IdxList& idxs, int idx1, int idx2, int wing_cnt, uint16_t cnt, CardVector& base, vector<CardVector>& res) {
    IdxList candidate;
    int n = idxs.size(), idx;
    for(int i = 0; i < n; i++) {
        idx = idxs[i];
        if(idx < idx1 || idx > idx2) candidate.push_back(idx);// 跳过区间[idx1,idx2]
    }
    if(candidate.size()) combine(candidate, wing_cnt, cnt, base, res);
}

int get_valid_size(IdxList& idxs, int bound) {
    int n = idxs.size();
    while(n && idxs[n-1] >= bound) n--;// 最多减三次，没必要二分搜索
    return n;
}

#define EQUAL(vec, i, val) (vec[(i)] - (i) == (val))

// idxs升序 在区间[left,right)中找到顺子断裂的位置
// 返回顺子的结束位置(不能取到)
int straight_break(IdxList& idxs, int left, int right, int val) {
    if(left == right) return left;
    if(right - left >= 8) {
        int mid;
        right--;
        while(left < right) {
            mid = (left + right) >> 1;
            if(EQUAL(idxs, mid, val)) left = mid + 1;
            else right = mid - 1;
        }
        return EQUAL(idxs, left, val) ? left+1 : left;
    }
    else while(left < right && EQUAL(idxs, left, val)) left++;
    return left;
}

// https://wiki.botzone.org.cn/index.php?title=FightTheLandlord
vector<CardVector> get_actions(CardVector& cards, CardVector& rival_cards) {
    static uint16_t straight[5] = {0,5,3,2,2};// 顺子的最短长度
    uint16_t rival_cnt = 0;
    vector<CardVector> res;
    vector<IdxList> cnt_idxs(5);// cnt_idxs[k]表示自己至少有k张的牌索引列表
    for(int i = 0; i < SIZE; i++) {
        for(int k = cards[i]; k; k--) cnt_idxs[k].push_back(i);
        rival_cnt += rival_cards[i];
    }
    if(rival_cnt == 0) {// 主动出牌，除过牌之外的任意合规牌型
        add_rocket(cards, res);
        // 单个，一对，三不带，炸弹
        auto end_it = cnt_idxs[1].end();
        for(auto it = cnt_idxs[1].begin(); it != end_it; it++) {
            int i = *it;
            for(uint16_t k = cards[i]; k; k--) {
                add_single_rank(i, k, res);
            }
        }
        // 三带一单，三带一对
        CardVector temp(SIZE, 0);
        end_it = cnt_idxs[3].end();
        for(auto it = cnt_idxs[3].begin(); it != end_it; it++) {
            int i = *it;
            temp[i] = 3;
            add_single_wing(cnt_idxs[1], i, i, 1, temp, res);
            add_single_wing(cnt_idxs[2], i, i, 2, temp, res);
            temp[i] = 0;
        }
        // 四带二单，四带二对
        end_it = cnt_idxs[4].end();
        for(auto it = cnt_idxs[4].begin(); it != end_it; it++) {
            int i = *it;
            for(uint16_t k = 1; k <= 2; k++) {
                IdxList& idxs = cnt_idxs[k];
                int n = idxs.size();
                for(int j1 = 0; j1 < n-1; j1++) {
                    int t1 = idxs[j1];
                    if(t1 == i) continue;
                    for(int j2 = j1+1; j2 < n; j2++) {
                        int t2 = idxs[j2];
                        if(t2 == i) continue;
                        CardVector temp(SIZE, 0);
                        temp[i] = 4;temp[t1] = temp[t2] = k;
                        res.emplace_back(move(temp));
                    }
                }
            }
        }
        // 单顺，双顺，三顺，四顺，三顺带同数量的单个或一对，四顺带同数量的两单或两对
        for(uint16_t k = 1; k <= 4; k++) {
            IdxList& idxs = cnt_idxs[k];
            int n = get_valid_size(idxs, TWO_IDX), min_len = straight[k];
            for(int i = 0, j = min_len; j <= n; j = i + min_len) {
                int val = idxs[i] - i;
                if(!EQUAL(idxs, j-1, val)) {
                    i++;
                    while(EQUAL(idxs, i, val)) i++;// 找到断裂位置
                    continue;
                }
                j = straight_break(idxs, j, n, val);
                int left = idxs[i], right = idxs[j-1], i1, i2, t;
                for(i1 = left, t = left+min_len-1; t <= right; i1++, t++) {
                    CardVector temp(SIZE, 0);
                    for(i2 = i1; i2 < t; i2++) temp[i2] = k;
                    while(i2 <= right) {
                        temp[i2++] = k;
                        res.push_back(temp);// 顺子区间[i1,i2)
                        if(k < 3) continue;
                        // 三顺带单个或一对，四顺带两单或两对
                        uint16_t factor = k == 3 ? 1 : 2;
                        uint16_t wing_cnt = (i2 - i1) * factor;// 翼的种数
                        for(uint16_t k1 = 1, temp_cnt = wing_cnt+(i2-i1); k1 <= 2; k1++) {// 每种翼的张数
                            if(cnt_idxs[k1].size() < temp_cnt) break;// 不能带单，必定不能带双
                            add_multi_wing(cnt_idxs[k1], i1, i2-1, wing_cnt, k1, temp, res);
                        }
                    }
                }
                i = j;
            }
        }
    }
    else {// 被动出牌，除过牌、炸弹外，牌型要相同
        res.emplace_back(SIZE, 0);// 过牌
        if(rival_cnt == 2 && rival_cards[BJ_IDX] && rival_cards[RJ_IDX]) return res;
        add_rocket(cards, res);
        uint16_t card_cnt = 0;// 自己有多少张牌
        uint16_t unique_cnt = 0;// 牌的种数
        int max_card = 0;// 最大牌对应的索引
        uint16_t max_cnt = 0;// 张数最大值
        int max_cnt_cnt = 0;// 张数等于max_cnt的牌有多少种
        int max_cnt_min = SIZE;// 张数等于max_cnt的最小牌的索引
        for(int i = 0; i < SIZE; i++) {
            if(rival_cards[i]) {
                unique_cnt++;
                max_card = i;
                if(rival_cards[i] > max_cnt) max_cnt = rival_cards[i];
            }
            card_cnt += cards[i];
        }
        IdxList& idxs = cnt_idxs[max_cnt];
        if(unique_cnt == 1) {// 只有一种牌
            int n = idxs.size(), i = 0;
            while(i < n && idxs[i] <= max_card) i++;
            while(i < n) add_single_rank(idxs[i++], max_cnt, res);
            if(max_cnt != 4) add_bomb(cnt_idxs[4], res);
            return res;
        }
        add_bomb(cnt_idxs[4], res);
        if(card_cnt < rival_cnt) return res;
        for(int i = 0; i < SIZE; i++) {
            if(rival_cards[i] == max_cnt) {
                max_cnt_cnt++;
                if(i < max_cnt_min) max_cnt_min = i;
            }
        }
        int end_idx = max_cnt_cnt == 1 ? TWO_IDX+1 : TWO_IDX;// 类似顺子的牌型不能取到2
        int n = get_valid_size(idxs, end_idx);
        int i = upper_bound(idxs.begin(), idxs.end(), max_cnt_min) - idxs.begin();
        if(i + max_cnt_cnt > n) return res;// 主牌张数不够
        uint16_t wing_cnt = 0, k = 0;// 翼的种数，每种翼的张数
        if(max_cnt > 2) {// 三顺、四顺才能带翼
            wing_cnt = unique_cnt - max_cnt_cnt;
            if(wing_cnt) k = (rival_cnt - max_cnt*max_cnt_cnt) / wing_cnt;
        }
        IdxList& idxs_k = cnt_idxs[k];
        if(wing_cnt && idxs_k.size() - max_cnt_cnt < wing_cnt) return res;// 张数不够带翼
        for(int j = i + max_cnt_cnt; j <= n; j = i + max_cnt_cnt) {
            int val = idxs[i] - i;
            if(!EQUAL(idxs, j-1, val)) {
                i = straight_break(idxs, i+1, j-1, val);
                continue;
            }
            j = straight_break(idxs, j, n, val);
            CardVector temp(SIZE, 0);
            int left = idxs[i], right = idxs[j-1];
            int i1 = left, t = left + max_cnt_cnt - 1;
            while(i1 < t) temp[i1++] = max_cnt;
            for(i1 = left; t <= right; i1++, t++) {// 固定长度的区间[i1,t]
                temp[t] = max_cnt;
                if(wing_cnt == 1) add_single_wing(idxs_k, i1, t, k, temp, res);
                else if(wing_cnt) add_multi_wing(idxs_k, i1, t, wing_cnt, k, temp, res);
                else res.push_back(temp);
                temp[i1] = 0;
            }
            i = j;
        }
    }
    return res;
}

inline int randint(int high) {
    return torch::randint(high, {}, torch::kInt16).item<int>();
}

inline double randd() {
    return torch::rand({}, torch::kDouble).item<double>();
}

int epsilon_greedy(const at::Tensor &q, double epsilon) {
    int len = q.size(0);
    if (len == 1) return 0;
    if (epsilon >= 1 || (epsilon > 0 && randd() < epsilon)) {
        return randint(len);
    }
    else return q.argmax().item<int>();
}

int RandomAgent::act(const Observation& obs) {
    return randint(obs.valid_moves.size());
}

DeepAgent::DeepAgent(const vector<string>& model_paths, bool jit, const string& device, double epsilon):jit(jit),epsilon(epsilon),device(c10::Device(device)) {
    assert(model_paths.size() == PLAYER_CNT);
    int lstm_input = 162, lstm_hidden = 128;
    int dim[PLAYER_CNT] = { 373, 484, 484 };
    for(int p = 0; p < PLAYER_CNT; p++) {
        if(jit) {
            jit_models.push_back(torch::jit::load(model_paths[p]));
            jit_models[p].to(this->device);
            jit_models[p].train(false);
        }
        else {
            models.emplace_back(lstm_input, lstm_hidden, dim[p] + lstm_hidden);
            torch::load(models[p], model_paths[p]);
            models[p]->to(this->device);
            models[p]->train(false);
        }
        if(p > 0) name += '_';
        name += model_paths[p];
    }
    if(epsilon > 0) {
        name += '_';
        name += to_string(epsilon);
    }
}
int DeepAgent::act(const Observation& obs) {
    if(obs.valid_moves.size() == 1) return 0;
    vector<at::Tensor> feature = get_feature(obs, device);
    torch::NoGradGuard no_grad;
    at::Tensor q;
    if(jit) q = jit_models[obs.player].forward({feature[3],feature[2]}).toTensor();
    else q = models[obs.player]->forward(feature[3],feature[2]);
    return epsilon_greedy(q, epsilon);
}

void delimiter(char c='#', int num=40) {
    while((num--)) cout << c;
    cout << endl;
}
bool vec_equal(const CardVector& a, const CardVector& b) {
    int size = a.size();
    if(b.size() != size) return false;
    for(int i = 0; i < size; i++) if(a[i] != b[i]) return false;
    return true;
}
int parse_action(const string& line, const vector<CardVector>& valid_moves) {
    // cout << line << endl;
    // cout << line.size() << endl;
    int size = valid_moves.size();
    CardVector action(SIZE, 0);
    char c;
    for(int i = line.size()-1; i >= 0; i--) {
        c = line[i];
        // cout << c << endl;
        // cout << (int)c << endl;
        if(c == ' ' || c == '\n' || c == '\r') continue;
        auto it = CARD_IDX.find(c);
        if(it == CARD_IDX.end()) return -1;
        // cout << it->first << ',' << it->second << endl;
        action[it->second]++;
    }
    for(int i = 0; i < size; i++)
        if(vec_equal(action, valid_moves[i])) return i;
    return -1;
}
int HumanAgent::act(const Observation& obs) {
    if(obs.valid_moves.empty()) return -1;
    delimiter();
    for(int p = 0; p < PLAYER_CNT; p++) {
        string str;
        if(p != obs.player) {
            parse_vec(obs.played_cards[p], str);
            cout << PLAYER_STR[p] << ':' << obs.player_cards_cnt[p] << " played:" << str << endl;
        }
        else {
            parse_vec(obs.cards, str);
            cout << PLAYER_STR[p] << ':' << str << endl;
        }
    }
    delimiter();
    cout << "input(" << obs.player << "):";
    int idx = -1;
    string line;
    while(idx == -1) {
        fflush(stdin);
        getline(cin, line);
        if(!cin.good()) {
            cin.clear();
            cout << endl;
            break;
        }
        idx = parse_action(line, obs.valid_moves);
        if(idx != -1) break;
        cout << "invalid:";
    }
    return idx;
}

using at::indexing::Slice;

at::Tensor cards2tensor(const CardVector& cards) {
    at::Tensor t = torch::zeros({ CARD_CNT }, torch::kInt16);
    uint16_t start;
    for (uint16_t i = 0; i < BJ_IDX; i++) {
        start = i << 2;
        t.index({ Slice(start, start + cards[i]) }) = 1;
    }
    if (cards[BJ_IDX]) t[CARD_CNT - 2] = 1;
    if (cards[RJ_IDX]) t[CARD_CNT - 1] = 1;
    return t;
}

at::Tensor one_hot(int index, int num_class) {
    at::Tensor t = torch::zeros({ num_class }, torch::kInt16);
    t[index] = 1;
    return t;
}

vector<at::Tensor> get_feature(const Observation& obs, const c10::Device& device) {
    int num_action = obs.valid_moves.size();
    at::Tensor cards = cards2tensor(obs.cards);
    at::Tensor other_cards = cards2tensor(obs.other_cards);
    at::Tensor last_move = cards2tensor(obs.last_move);
    at::Tensor valid_moves = torch::zeros({num_action, CARD_CNT}, torch::kInt16);
    for(int i = 0; i < num_action; i++) {
        valid_moves[i] = cards2tensor(obs.valid_moves[i]);
    }
    at::Tensor bomb_num = one_hot(obs.bomb_num, 15);// 炸弹最多有14个
    at::Tensor x;
    if(obs.player == 0) {// 地主
        int prev = prev_player(0), next = next_player(0);
        at::Tensor prev_cards_cnt = one_hot(obs.player_cards_cnt[prev]-1, 17);
        at::Tensor prev_played_cards = cards2tensor(obs.played_cards[prev]);
        at::Tensor next_cards_cnt = one_hot(obs.player_cards_cnt[next]-1, 17);
        at::Tensor next_played_cards = cards2tensor(obs.played_cards[next]);
        x = torch::cat({cards,
                        other_cards,
                        last_move,
                        prev_played_cards,
                        next_played_cards,
                        prev_cards_cnt,
                        next_cards_cnt,
                        bomb_num});
    }
    else {
        int teammate = obs.player == 2 ? 1 : 2;
        at::Tensor landlord_played_cards = cards2tensor(obs.played_cards[0]);
        at::Tensor teammate_played_cards = cards2tensor(obs.played_cards[teammate]);
        at::Tensor landlord_last_move = cards2tensor(obs.last_moves[0]);
        at::Tensor teammate_last_move = cards2tensor(obs.last_moves[teammate]);
        at::Tensor landlord_cards_cnt = one_hot(obs.player_cards_cnt[0]-1, 20);
        at::Tensor teammate_cards_cnt = one_hot(obs.player_cards_cnt[teammate]-1, 17);
        x = torch::cat({cards,
                        other_cards,
                        landlord_played_cards,
                        teammate_played_cards,
                        last_move,
                        landlord_last_move,
                        teammate_last_move,
                        landlord_cards_cnt,
                        teammate_cards_cnt,
                        bomb_num});
    }
    x = x.to(device);
    valid_moves = valid_moves.to(device);
    at::Tensor x_batch = x.repeat({ num_action,1 });
    x_batch = torch::cat({x_batch,valid_moves}, 1);
    int length = 15, size = obs.history.size();
    at::Tensor z = torch::zeros({length, CARD_CNT}, torch::kInt16);
    int i = 0, j = size - length;
    if(j < 0) {
        i = -j;
        j = 0;
    }
    for(; i < length; i++, j++) z[i] = cards2tensor(obs.history[j]);
    z = z.reshape({5, 162}).to(device);
    at::Tensor z_batch = z.repeat({num_action,1,1});
    return {x, z, x_batch.to(torch::kF32), z_batch.to(torch::kF32)};
}

void list2vec(const vector<uint16_t>& cards, int start, int stop, CardVector& vec) {
    vec.assign(SIZE, 0);
    for(int i = start; i < stop; i++) vec[cards[i]]++;
}
void parse_vec(const CardVector& vec, string& str) {
    str = "[";
    char c;
    for(int i = 0; i < SIZE; i++) {
        c = CARD_CHAR[i];
        for(int j = vec[i]; j > 0; j--) { str += c;str += ' '; }
    }
    if(str.size() == 1) str += ']';
    else str.back() = ']';
}

void vec_add(CardVector& out, CardVector& a, CardVector& b) {
    for(int i = 0; i < SIZE; i++) out[i] = a[i] + b[i];
}
void vec_minus(CardVector& out, CardVector& a, CardVector& b) {
    for(int i = 0; i < SIZE; i++) out[i] = a[i] - b[i];
}

Game::Game(const string& objective, vector<shared_ptr<Agent>>& players, uint16_t info):
    objective(objective),players(players),info(info),player_cards(PLAYER_CNT),init_cards(PLAYER_CNT) {
    assert(objective == "wp" || objective == "adp" || objective == "logadp");
    assert(players.size() == PLAYER_CNT);
    init_deck();
    obs.other_cards = CardVector(SIZE, 0);
}
void Game::init_deck() {// 不区分suit,只考虑rank
    for(int i = 0; i < BJ_IDX; i++) {
        for(int j = 0; j < 4; j++) deck.push_back(i);
    }
    deck.push_back(BJ_IDX);
    deck.push_back(RJ_IDX);
}
const Observation& Game::reset(const vector<shared_ptr<Agent>>& players, const vector<uint16_t>& deck) {
    int size = players.size();
    if(size > 0) {
        assert(size == PLAYER_CNT);
        this->players = players;
    }
    size = deck.size();
    if(size > 0) {
        assert(size == CARD_CNT);
        uint16_t min_idx = SIZE, max_idx = 0;
        for(int i = 0; i < CARD_CNT; i++) {
            min_idx = min(min_idx, deck[i]);
            max_idx = max(max_idx, deck[i]);
        }
        assert(min_idx == 0 && max_idx == RJ_IDX);
        CardVector vec;
        list2vec(deck, 0, size, vec);
        for(int i = 0; i < BJ_IDX; i++) assert(vec[i] == 4);
        assert(vec[BJ_IDX] == 1 && vec[RJ_IDX] == 1);
        this->deck = deck;
    }
    else {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        shuffle(this->deck.begin(), this->deck.end(), std::default_random_engine(seed));
    }
    over = false;
    obs.history.clear();
    obs.player = 0;
    obs.last_moves = vector<CardVector>(PLAYER_CNT, CardVector(SIZE, 0));
    obs.played_cards = vector<CardVector>(PLAYER_CNT, CardVector(SIZE, 0));
    obs.last_move = CardVector(SIZE,0);
    obs.bomb_num = 0;
    deal_card();
    return get_obs();
}
void Game::deal_card() {
    vector<uint16_t>& player_cards_cnt = obs.player_cards_cnt;
    player_cards_cnt = {20, 17, 17};
    int i = 0;
    for(int p = 0; p < PLAYER_CNT; p++) {
        list2vec(deck, i, i+player_cards_cnt[p], player_cards[p]);
        init_cards[p] = player_cards[p];
        i += player_cards_cnt[p];
    }
    obs.three_cards.assign(deck.begin(), deck.begin()+3);
    obs.valid_moves = get_actions(player_cards[0], obs.last_move);
    if(info == 2) {
        string str;
        for(int p = 0; p < PLAYER_CNT; p++) {
            parse_vec(player_cards[p], str);
            cout << PLAYER_STR[p] << ':' << str << endl;
        }
    }
}
const Observation& Game::get_obs() {
    obs.done = over;
    obs.reward = get_reward();
    obs.cards = player_cards[obs.player];
    uint16_t next = next_player(obs.player), prev = prev_player(obs.player);
    vec_add(obs.other_cards, player_cards[next], player_cards[prev]);
    return obs;
}
float Game::get_reward() {// 地主视角的奖励
    if(!over) return 0.0;
    float reward = 1.0;
    if(objective == "adp") reward = 1 << obs.bomb_num;
    else if(objective == "logadp") reward = 1 + obs.bomb_num;
    if(obs.player != 0) return -reward;
    return reward;
}
int Game::get_winner() {// 最先出完牌的玩家索引
    if(!over) return -1;
    return obs.player;
}
bool Game::is_over() {
    return over;
}
uint16_t Game::get_bomb_num() {
    return obs.bomb_num;
}
const Observation& Game::step(int action_idx) {
    assert(!over);
    uint16_t p = obs.player;
    if(action_idx == -1) action_idx = players[p]->act(obs);
    assert(action_idx >= 0 && action_idx < obs.valid_moves.size());
    CardVector action = obs.valid_moves[action_idx];
    if(info > 0) {
        string str;
        parse_vec(action, str);
        cout << PLAYER_STR[p] << ':' << str << endl;
    }
    uint16_t action_cnt = 0, max_cnt = 0;
    for(int i = 0; i < SIZE; i++) {
        action_cnt += action[i];
        max_cnt = max(max_cnt, action[i]);
    }
    if(action_cnt > 0) {
        vec_add(obs.played_cards[p], obs.played_cards[p], action);
        vec_minus(player_cards[p], player_cards[p], action);
        obs.player_cards_cnt[p] -= action_cnt;
        if((action_cnt == 2 && action[BJ_IDX] && action[RJ_IDX]) || (action_cnt == 4 && max_cnt == 4))
            obs.bomb_num += 1;
        obs.last_move = action;
    }
    else obs.last_move = obs.history.back();
    obs.history.emplace_back(move(action));
    if(obs.player_cards_cnt[p] == 0) {
        over = true;
        obs.valid_moves.clear();
    }
    else {
        obs.player = next_player(p);
        obs.valid_moves = get_actions(player_cards[obs.player], obs.last_move);
    }
    return get_obs();
}
GameData Game::game_data() {
    GameData data;
    data.winner = get_winner();
    data.deck = deck;
    for(int p = 0; p < PLAYER_CNT; p++)
        data.players.push_back(players[p]->name);
    data.history = obs.history;
    data.init_cards = init_cards;
    return data;
}

void parse_game_data(size_t id, const GameData& data, const string& save_path) {
    assert(data.deck.size() == CARD_CNT && data.players.size() == PLAYER_CNT);
    ofstream file(save_path, ios::app);
    file << "Game " << id << endl;
    for (uint16_t p = 0; p < PLAYER_CNT; p++)
        file << PLAYER_STR[p] << ':' << data.players[p] << endl;
    file << "winner:" << data.winner << endl;
    string str;
    for (uint16_t p = 0; p < PLAYER_CNT; p++) {
        parse_vec(data.init_cards[p], str);
        file << PLAYER_STR[p] << ':' << str << endl;
    }
    const vector<CardVector>& history = data.history;
    uint16_t n = history.size();
    for (uint16_t i = 0, p = 0; i < n; i++) {
        parse_vec(history[i], str);
        file << PLAYER_STR[p] << ':' << str << endl;
        p = next_player(p);
    }
    file << endl;
    file.close();
}