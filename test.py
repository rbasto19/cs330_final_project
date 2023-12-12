import os
import argparse
import pickle
import yaml
import torch
from glob import glob
from tqdm.auto import tqdm
from easydict import EasyDict

from models.epsnet import *
from utils.datasets import *
from utils.transforms import *
from utils.misc import *
import higher
import random
from utils.common import get_optimizer, get_scheduler
from torch_geometric.data import Batch
from torch_geometric.data import DataLoader

def num_confs(num:str):
    if num.endswith('x'):
        return lambda x:x*int(num[:-1])
    elif int(num) > 0: 
        return lambda x:int(num)
    else:
        raise ValueError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', type=str, help='path for loading the checkpoint')
    parser.add_argument('--save_traj', action='store_true', default=False,
                    help='whether store the whole trajectory for sampling')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--num_confs', type=num_confs, default=num_confs('2x'))
    parser.add_argument('--test_set', type=str, default=None)
    parser.add_argument('--start_idx', type=int, default=800)
    parser.add_argument('--end_idx', type=int, default=1000)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--clip', type=float, default=1000.0)
    parser.add_argument('--n_steps', type=int, default=5000,
                    help='sampling num steps; for DSM framework, this means num steps for each noise scale')
    parser.add_argument('--global_start_sigma', type=float, default=0.5,
                    help='enable global gradients only when noise is low')
    parser.add_argument('--w_global', type=float, default=1.0,
                    help='weight for global gradients')
    # Parameters for DDPM
    parser.add_argument('--sampling_type', type=str, default='ld',
                    help='generalized, ddpm_noisy, ld: sampling method for DDIM, DDPM or Langevin Dynamics')
    parser.add_argument('--eta', type=float, default=1.0,
                    help='weight for DDIM and DDPM: 0->DDIM, 1->DDPM')
    parser.add_argument('--MAML', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="drugs")
    args = parser.parse_args()
    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=args.device)
    #config_path = glob(os.path.join(os.path.dirname(os.path.dirname(args.ckpt)), '*.yml'))[0]
    config_path = "configs/"+args.dataset+"_default.yml"
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    seed_all(config.train.seed)
    log_dir = os.path.dirname(os.path.dirname(args.ckpt))
    
    # Logging
    output_dir = get_new_log_dir(log_dir, 'sample', tag=args.tag)
    logger = get_logger('test', output_dir)
    logger.info(args)

    # Datasets and loaders
    logger.info('Loading datasets...')
    transforms = Compose([
        CountNodesPerGraph(),
        AddHigherOrderEdges(order=config.model.edge_order), # Offline edge augmentation
    ])
    if args.test_set is None:
        test_set = PackedConformationDataset(config.dataset.test, transform=transforms)
    else:
        test_set = PackedConformationDataset(args.test_set, transform=transforms)
    print(config.dataset.test)
    # Model
    logger.info('Loading model...')
    model = get_model(ckpt['config'].model,lr_init=config.train.optimizer.lr).to(args.device)
    model.load_state_dict(ckpt['model'])
    #don't have to reinitialize optimizer from checkpoint bc fine tuning, so different optimizer
    #so just use general configs for qm9_default. 
    #If bad check this out.
    config_path = './configs/qm9_default.yml'
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    optimizer = get_optimizer(config.train.optimizer, list(model.model_global.parameters()) + list(model.model_local.parameters()))
    test_set_selected = []
    for i, data in enumerate(test_set):
        if not (args.start_idx <= i < args.end_idx): continue
        test_set_selected.append(data)
    
    done_smiles = set()
    results = []
    if args.resume is not None:
        with open(args.resume, 'rb') as f:
            results = pickle.load(f)
        for data in results:
            done_smiles.add(data.smiles)
    #these below are the parameters with best performance from tensorboard plots
    size_task = 5
    size_support = 4
    size_query = size_task - size_support
    num_inner_loop_steps = 2

    for i, data in enumerate(tqdm(test_set_selected)):
        if data.smiles in done_smiles:
            logger.info('Molecule#%d is already done.' % i)
            continue

        num_refs = data.pos_ref.size(0) // data.num_nodes
        num_samples = args.num_confs(num_refs)
        
        data_input = data.clone()
        data_input['pos_ref'] = None
        batch = repeat_data(data_input, num_samples).to(args.device)
        
        clip_local = None
        if len(test_set._packed_data[data.smiles]) < 5 and args.MAML == True:
            logger.info('not enough known conformations for MAML setup')
            continue
        
        task = random.sample(test_set._packed_data[data.smiles], size_task)
        support = transforms(Batch.from_data_list(task[:size_support])).to(args.device)  #needs to be a batch object with only support set
        query = transforms(Batch.from_data_list(task[size_support:])).to(args.device)  #also a batch object with only query set
        for _ in range(2):  # Maximum number of retry
            try:
                pos_init = torch.randn(batch.num_nodes, 3).to(args.device)
                if args.MAML == True:
                    #have to do fine-tuning: do inner loop like in checkpoint model, then compute things
                    optimizer.zero_grad()
                    with higher.innerloop_ctx(model, optimizer, copy_initial_weights=False) as (fnet, diffopt):
                        for _ in range(num_inner_loop_steps):
                            support_loss, loss_global, loss_local = fnet.get_loss(
                                atom_type=support.atom_type,
                                pos=support.pos,
                                bond_index=support.edge_index,
                                bond_type=support.edge_type,
                                batch=support.batch,
                                num_nodes_per_graph=support.num_nodes_per_graph,
                                num_graphs=support.num_graphs,
                                anneal_power=config.train.anneal_power,
                                return_unreduced_loss=True
                            )
                            support_loss = support_loss.mean()
                            #print("support_loss mean:", support_loss)
                            diffopt.step(support_loss)
                        pos_gen, pos_gen_traj = fnet.langevin_dynamics_sample( #difference is fnet instead of model. batch already accounts for number of necessary generations
                            atom_type=batch.atom_type,
                            pos_init=pos_init,
                            bond_index=batch.edge_index,
                            bond_type=batch.edge_type,
                            batch=batch.batch,
                            num_graphs=batch.num_graphs,
                            extend_order=False, # Done in transforms.
                            n_steps=args.n_steps,
                            step_lr=1e-6,
                            w_global=args.w_global,
                            global_start_sigma=args.global_start_sigma,
                            clip=args.clip,
                            clip_local=clip_local,
                            sampling_type=args.sampling_type,
                            eta=args.eta
                        )
                else:
                    pos_gen, pos_gen_traj = model.langevin_dynamics_sample(
                        atom_type=batch.atom_type,
                        pos_init=pos_init,
                        bond_index=batch.edge_index,
                        bond_type=batch.edge_type,
                        batch=batch.batch,
                        num_graphs=batch.num_graphs,
                        extend_order=False, # Done in transforms.
                        n_steps=args.n_steps,
                        step_lr=1e-6,
                        w_global=args.w_global,
                        global_start_sigma=args.global_start_sigma,
                        clip=args.clip,
                        clip_local=clip_local,
                        sampling_type=args.sampling_type,
                        eta=args.eta
                    )
                pos_gen = pos_gen.cpu()
                if args.save_traj:
                    data.pos_gen = torch.stack(pos_gen_traj)
                else:
                    data.pos_gen = pos_gen
                results.append(data)
                done_smiles.add(data.smiles)

                save_path = os.path.join(output_dir, 'samples_%d.pkl' % i)
                logger.info('Saving samples to: %s' % save_path)
                with open(save_path, 'wb') as f:
                    pickle.dump(results, f)

                break   # No errors occured, break the retry loop
            except FloatingPointError:
                clip_local = 20
                logger.warning('Retrying with local clipping.')

    save_path = os.path.join(output_dir, 'samples_all.pkl')
    logger.info('Saving samples to: %s' % save_path)

    def get_mol_key(data):
        for i, d in enumerate(test_set_selected):
            if d.smiles == data.smiles:
                return i
        return -1
    results.sort(key=get_mol_key)

    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
        
    