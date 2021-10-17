#if !defined(_MODEL_LOCKER_H_)
#define _MODEL_LOCKER_H_

#include <mutex>
#include <cassert>
#include "model.h"
#include "game.h"
using namespace std;

using StateDict = torch::OrderedDict<string, at::Tensor>;

void load_state_dict(const StateDict& src_params, const StateDict& src_buffers, LstmModel& dst_module) {
	torch::NoGradGuard no_grad;
	StateDict dst_params = dst_module->named_parameters(true);
	StateDict dst_buffers = dst_module->named_buffers(true);
	TORCH_CHECK(src_params.size() == dst_params.size(), "The number of parameters of two modules are not equal.");
	TORCH_CHECK(src_buffers.size() == dst_buffers.size(), "The number of buffers of two modules are not equal.");
	for (auto& dst : dst_params) {
		const string& name = dst.key();
		const at::Tensor* src = src_params.find(name);
		TORCH_CHECK(src != NULL, "original module doesn't have the " + name + " parameter.");
		dst.value().copy_(*src);
	}
	for (auto& dst : dst_buffers) {
		const string& name = dst.key();
		const at::Tensor* src = src_buffers.find(name);
		TORCH_CHECK(src != NULL, "original module doesn't have the " + name + " buffer.");
		dst.value().copy_(*src);
	}
}
void load_state_dict(const LstmModel& src_module, LstmModel& dst_module) {
    StateDict src_params = src_module->named_parameters(true);
    StateDict src_buffers = src_module->named_buffers(true);
    load_state_dict(src_params, src_buffers, dst_module);
}

class ModelLocker {// 3种模型加锁
public:
    ModelLocker(vector<LstmModel>& models):models(models),locks(PLAYER_CNT) {
        uint16_t size = models.size();
        assert(size == PLAYER_CNT);
    }
    void update(uint16_t p, const LstmModel& model) {// 训练线程调用
        locks[p].lock();
        load_state_dict(model, models[p]);
        locks[p].unlock();
    }
    void update(uint16_t p, const StateDict& params, const StateDict& buffers) {// 训练线程调用
        locks[p].lock();
        load_state_dict(params, buffers, models[p]);
        locks[p].unlock();
    }
    at::Tensor forward(uint16_t p, at::Tensor& z, at::Tensor& x) {
        torch::NoGradGuard no_grad;
        locks[p].lock();
        at::Tensor output = models[p]->forward(z, x);
        locks[p].unlock();
        return output;
    }
private:
    vector<mutex> locks;
    vector<LstmModel>& models;
};

#endif // _MODEL_LOCKER_H_
