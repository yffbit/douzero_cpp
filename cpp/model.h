#if !defined(_MODEL_H_)
#define _MODEL_H_

#include <string>
#include <torch/torch.h>
using std::string;
using std::tuple;

struct LstmModelImpl : torch::nn::Module {
	LstmModelImpl(int lstm_input, int lstm_hidden, int fc_input, int fc_hidden = 512) :
		lstm(torch::nn::LSTMOptions(lstm_input, lstm_hidden).batch_first(true)),
		dense1(fc_input, fc_hidden),
		dense2(fc_hidden, fc_hidden),
		dense3(fc_hidden, fc_hidden),
		dense4(fc_hidden, fc_hidden),
		dense5(fc_hidden, fc_hidden),
		dense6(fc_hidden, 1) {
		register_module("lstm", lstm);
		register_module("dense1", dense1);
		register_module("dense2", dense2);
		register_module("dense3", dense3);
		register_module("dense4", dense4);
		register_module("dense5", dense5);
		register_module("dense6", dense6);
	}
	at::Tensor forward(at::Tensor& z, at::Tensor& x) {
		tuple<at::Tensor, tuple<at::Tensor, at::Tensor>> lstm_out = lstm(z);
		at::Tensor temp = std::get<0>(lstm_out);
		temp = temp.select(1, temp.size(1) - 1);// 最后一个时间步
		temp = torch::cat({ temp,x }, 1);
		temp = dense1(temp).relu();
		temp = dense2(temp).relu();
		temp = dense3(temp).relu();
		temp = dense4(temp).relu();
		temp = dense5(temp).relu();
		return dense6(temp);
	}
	torch::nn::LSTM lstm;
	torch::nn::Linear dense1, dense2, dense3, dense4, dense5, dense6;
};
TORCH_MODULE(LstmModel);

#endif // _MODEL_H_
