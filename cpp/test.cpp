#include "game.h"

void game_loop(Game& game) {
    game.reset();
    while(!game.is_over()) {
        game.step();
    }
    cout << endl;
}

int main() {
    try {
        vector<string> model_paths = {
            "D:/Project/DouZero/baselines/douzero_WP/landlord.ckpt.cupt",
            "D:/Project/DouZero/baselines/douzero_WP/landlord_down.ckpt.cupt",
            "D:/Project/DouZero/baselines/douzero_WP/landlord_up.ckpt.cupt"
        };
        vector<string> model_paths1 = {
            "D:/Project/DouZero/douzero_cpp/checkpoints/douzero_wp/cppmodel_0_441600.pt",
            "D:/Project/DouZero/douzero_cpp/checkpoints/douzero_wp/cppmodel_1_441600.pt",
            "D:/Project/DouZero/douzero_cpp/checkpoints/douzero_wp/cppmodel_2_428800.pt"
        };
        shared_ptr<Agent> agent1 = make_shared<DeepAgent>(model_paths, true);
        shared_ptr<Agent> agent2 = make_shared<DeepAgent>(model_paths, true, "cuda:0");
        shared_ptr<Agent> agent3 = make_shared<DeepAgent>(model_paths1, false);
        shared_ptr<Agent> agent4 = make_shared<DeepAgent>(model_paths1, false, "cuda:0");
        shared_ptr<Agent> agent5 = make_shared<HumanAgent>("yff");
        vector<shared_ptr<Agent>> players = {agent1, agent2, agent2};
        for(auto p : players) cout << p->name << endl;
        Game game("wp", players, 2);
        game_loop(game);
        players = {agent3, agent4, agent4};
        for(auto p : players) cout << p->name << endl;
        game.reset(players);
        game_loop(game);
        players = { agent2, agent4, agent5 };
        for (auto p : players) cout << p->name << endl;
        game.reset(players);
        game_loop(game);
    }
    catch(exception& e) {
        cout << e.what() << endl;
    }
    return 0;
}
