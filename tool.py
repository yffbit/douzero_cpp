from douzero.dmc.models import LandlordLstmModel, FarmerLstmModel
import torch

if __name__ == '__main__':
    # device = 'cuda:0'
    device = 'cpu'
    suffix = '.pt'
    if 'cuda' in device:
        suffix = '.cupt'
    weight_paths = ['baselines/douzero_WP/landlord.ckpt',
                    'baselines/douzero_WP/landlord_down.ckpt',
                    'baselines/douzero_WP/landlord_up.ckpt']
    landlord = LandlordLstmModel()
    landlord_down = FarmerLstmModel()
    landlord_up = FarmerLstmModel()
    models = [landlord, landlord_down, landlord_up]
    for i,model in enumerate(models):
        state_dict = torch.load(weight_paths[i], map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        # torch.save(model, weight_paths[i] + '.pt')
        script_model = torch.jit.script(model)
        script_model.save(weight_paths[i] + suffix)
