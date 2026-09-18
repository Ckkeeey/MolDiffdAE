import os
import sys
import argparse
import pickle
import numpy as np

sys.path.append('.')

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from tqdm import tqdm
from torch_geometric.data import Batch
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

from models.model_ae import MolDiffAE
from utils.dataset import get_dataset
from utils.transforms import FeaturizeMol, Compose
from utils.misc import *
from utils.reconstruct import *
from utils.sample import seperate_outputs_no_traj

from easydict import EasyDict
from multiprocessing import Process, Queue


def reconstruct_worker(q, mol_info):
    try:
        mol = reconstruct_from_generated_with_edges(
            mol_info,
            add_edge=None
        )
        q.put(mol)
    except Exception:
        q.put(None)


def process_outputs(
    outputs,
    batch_node_raw,
    halfedge_index_raw,
    batch_halfedge_raw,
    n_graphs=1
):
    batch_node = batch_node_raw.cpu().numpy()
    halfedge_index = halfedge_index_raw.cpu().numpy()
    batch_halfedge = batch_halfedge_raw.cpu().numpy()

    output_list = seperate_outputs_no_traj(
        outputs,
        n_graphs,
        batch_node,
        halfedge_index,
        batch_halfedge
    )

    gen_list = []

    for output_mol in output_list:

        try:
            mol_info = featurizer.decode_output(
                pred_node=output_mol['pred'][0],
                pred_pos=output_mol['pred'][1],
                pred_halfedge=output_mol['pred'][2],
                halfedge_index=output_mol['halfedge_index'],
            )
        except Exception as e:
            print('[WARN] decode_output failed:', e)
            continue

        q = Queue()

        p = Process(
            target=reconstruct_worker,
            args=(q, mol_info)
        )

        p.start()
        p.join(timeout=15.0)

        if p.exitcode is None:
            p.terminate()
            p.join()
            print('[WARN] reconstruction timeout')
            continue

        if q.empty():
            print('[WARN] reconstruction returned nothing')
            continue

        rdmol = q.get()

        if rdmol is None:
            print('[WARN] reconstruction failed')
            continue

        mol_info['rdmol'] = rdmol

        try:
            smiles = Chem.MolToSmiles(rdmol)
        except Exception as e:
            print('[WARN] MolToSmiles failed:', e)
            continue

        mol_info['smiles'] = smiles

        # 与原 multiobj sampler 保持一致：
        # 含 "." 的多片段分子不作为有效结果
        if '.' in smiles:
            print('[WARN] disconnected molecule:', smiles)
            continue

        gen_list.append(mol_info)

    return gen_list


@torch.no_grad()
def sample_template_ddim(
    data,
    model,
    noise='deterministic',
    stride=1,
    n_graphs=1,
    start_step=1000
):
    """
    Original/template-conditioned DDIM.

    deterministic:
        source molecule
        -> encode
        -> deterministic forward DDIM
        -> reverse DDIM

    random:
        source molecule
        -> encode
        -> random x_T
        -> reverse DDIM

    No property manipulation.
    No classifier guidance.
    """

    time_sequence = list(
        range(
            0,
            min(start_step, model.num_timesteps) - stride,
            stride
        )
    )

    batch = Batch.from_data_list(
        [data.clone() for _ in range(n_graphs)],
        follow_batch=['halfedge_type', 'node_type']
    ).to(device)

    node_type = batch.node_type
    node_pos = batch.node_pos
    batch_node = batch.node_type_batch

    halfedge_type = batch.halfedge_type
    halfedge_index = batch.halfedge_index
    batch_halfedge = batch.halfedge_type_batch

    num_mol = batch.num_graphs

    # =========================================================
    # 1. Encode source molecule
    # =========================================================
    emb = model.encode(
        node_type,
        node_pos,
        batch_node,
        halfedge_type,
        halfedge_index,
        batch_halfedge,
        num_mol
    )

    edge_index = torch.cat(
        [halfedge_index, halfedge_index.flip(0)],
        dim=1
    )

    batch_edge = torch.cat(
        [batch_halfedge, batch_halfedge],
        dim=0
    )

    # =========================================================
    # 2. Construct x_T
    # =========================================================
    if noise == 'deterministic':

        # Continuous MolDiffAE scaling = [1, 4, 8]
        h_node_pert = (
            torch.nn.functional.one_hot(
                batch.node_type,
                model.num_node_types
            ).float()
            / model.scaling[1]
        )

        pos_pert = (
            batch.node_pos
            / model.scaling[0]
        )

        h_halfedge_pert = (
            torch.nn.functional.one_hot(
                batch.halfedge_type,
                model.num_edge_types
            ).float()
            / model.scaling[2]
        )

        # 0 -> 1 -> ... -> 998
        for step in tqdm(
            time_sequence,
            total=len(time_sequence),
            desc='DDIM forward'
        ):

            time_step = torch.full(
                (n_graphs,),
                step,
                dtype=torch.long,
                device=device
            )

            h_edge_pert = torch.cat(
                [h_halfedge_pert, h_halfedge_pert],
                dim=0
            )

            preds = model(
                h_node_pert,
                pos_pert,
                batch_node,
                h_edge_pert,
                edge_index,
                batch_edge,
                time_step,
                emb
            )

            pred_node = preds['pred_node'].detach()
            pred_pos = preds['pred_pos'].detach()
            pred_halfedge = preds['pred_halfedge'].detach()

            pos_pert = model.pos_transition.reverse_sample_ddim(
                x_t=pos_pert,
                x_recon=pred_pos,
                t=time_step,
                s=time_step + stride,
                batch=batch_node
            )

            h_node_pert = model.node_transition.reverse_sample_ddim(
                x_t=h_node_pert,
                x_recon=pred_node,
                t=time_step,
                s=time_step + stride,
                batch=batch_node
            )

            h_halfedge_pert = model.edge_transition.reverse_sample_ddim(
                x_t=h_halfedge_pert,
                x_recon=pred_halfedge,
                t=time_step,
                s=time_step + stride,
                batch=batch_halfedge
            )

        pos_init = pos_pert
        h_node_init = h_node_pert
        h_halfedge_init = h_halfedge_pert

    elif noise == 'random':

        n_nodes_all = node_type.shape[0]
        n_halfedges_all = halfedge_type.shape[0]

        h_node_init = model.node_transition.sample_init(
            n_nodes_all
        )

        pos_init = model.pos_transition.sample_init(
            [n_nodes_all, 3]
        )

        h_halfedge_init = model.edge_transition.sample_init(
            n_halfedges_all
        )

    else:
        raise ValueError(
            "noise must be 'deterministic' or 'random'"
        )

    h_node_pert = h_node_init
    pos_pert = pos_init
    h_halfedge_pert = h_halfedge_init

    # =========================================================
    # 3. Reverse DDIM
    # =========================================================
    for step in tqdm(
        time_sequence[::-1],
        total=len(time_sequence),
        desc='DDIM reverse'
    ):

        time_step = torch.full(
            (n_graphs,),
            step,
            dtype=torch.long,
            device=device
        )

        h_edge_pert = torch.cat(
            [h_halfedge_pert, h_halfedge_pert],
            dim=0
        )

        preds = model(
            h_node_pert,
            pos_pert,
            batch_node,
            h_edge_pert,
            edge_index,
            batch_edge,
            time_step,
            emb
        )

        pred_node = preds['pred_node'].detach()
        pred_pos = preds['pred_pos'].detach()
        pred_halfedge = preds['pred_halfedge'].detach()

        pos_pert = model.pos_transition.reverse_sample_ddim(
            x_t=pos_pert,
            x_recon=pred_pos,
            t=time_step,
            s=time_step - stride,
            batch=batch_node
        )

        h_node_pert = model.node_transition.reverse_sample_ddim(
            x_t=h_node_pert,
            x_recon=pred_node,
            t=time_step,
            s=time_step - stride,
            batch=batch_node
        )

        h_halfedge_pert = model.edge_transition.reverse_sample_ddim(
            x_t=h_halfedge_pert,
            x_recon=pred_halfedge,
            t=time_step,
            s=time_step - stride,
            batch=batch_halfedge
        )

    # =========================================================
    # 4. Reconstruction
    # =========================================================
    outputs = {
        'pred': [
            pred_node,
            pred_pos,
            pred_halfedge
        ]
    }

    outputs = {
        key: [
            v.detach().cpu().numpy()
            for v in value
        ]
        for key, value in outputs.items()
    }

    return process_outputs(
        outputs,
        batch_node,
        halfedge_index,
        batch_halfedge,
        n_graphs=n_graphs
    )


def save_sdf(gen_dict, sdf_dir):
    os.makedirs(sdf_dir, exist_ok=True)

    total = 0

    for source_id, mol_list in gen_dict.items():

        for mol_info in mol_list:

            mol = mol_info.get('rdmol')

            if mol is None:
                continue

            path = os.path.join(
                sdf_dir,
                f'{total}.sdf'
            )

            try:
                mol.SetProp(
                    'source_id',
                    str(source_id)
                )

                mol.SetProp(
                    'smiles',
                    mol_info.get(
                        'smiles',
                        Chem.MolToSmiles(mol)
                    )
                )

                writer = Chem.SDWriter(path)
                writer.write(mol)
                writer.close()

                total += 1

            except Exception as e:
                print(
                    '[WARN] failed to save SDF:',
                    e
                )

    return total


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config',
        type=str,
        default='configs/train/train_MolDiffAE_continuous.yml'
    )

    parser.add_argument(
        '--name',
        type=str,
        default='drug3d'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0'
    )

    parser.add_argument(
        '--ckpt',
        type=str,
        required=True
    )

    parser.add_argument(
        '--noise',
        type=str,
        choices=['deterministic', 'random'],
        default='deterministic'
    )

    parser.add_argument(
        '--start_step',
        type=int,
        default=1000
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None
    )

    parser.add_argument(
        '--logdir',
        type=str,
        default='outputs'
    )

    args = parser.parse_args()

    global device
    device = args.device

    # =========================================================
    # 1. Load config / checkpoint
    # =========================================================
    config = load_config(args.config)

    ckpt = torch.load(
        args.ckpt,
        map_location=args.device
    )

    print('Checkpoint iteration:', ckpt.get('iteration'))

    # =========================================================
    # 2. Dataset
    # =========================================================
    seed_all(config.train.seed)

    featurizer = FeaturizeMol(
        config.chem.atomic_numbers,
        config.chem.mol_bond_types,
        use_mask_node=config.transform.use_mask_node,
        use_mask_edge=config.transform.use_mask_edge,
        random=False
    )

    transform = Compose([
        featurizer
    ])

    dataset, subsets = get_dataset(
        config=config.dataset,
        transform=transform
    )

    test_set = subsets['test']

    print('Test molecules:', len(test_set))

    # =========================================================
    # 3. Build model using checkpoint config
    # =========================================================
    ckpt_config = ckpt['config']

    model = MolDiffAE(
        config=ckpt_config.model,
        num_node_types=featurizer.num_node_types,
        num_edge_types=featurizer.num_edge_types
    ).to(args.device)

    print(
        'Trainable parameters:',
        np.sum([
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        ])
    )

    model.load_state_dict(
        ckpt['model'],
        strict=False
    )

    model.eval()

    print('Model loaded successfully.')
    print('num_timesteps:', model.num_timesteps)
    print('scaling:', model.scaling)

    # =========================================================
    # 4. Output names
    # =========================================================
    if args.noise == 'deterministic':
        save_name = (
            f'template-{args.name}'
            f'_ddim-{args.start_step - 1}'
            f'-deterministic_test'
        )
    else:
        save_name = (
            f'template-{args.name}'
            f'_ddim-{args.start_step - 1}'
            f'-random_test'
        )

    os.makedirs(args.logdir, exist_ok=True)

    pkl_path = os.path.join(
        args.logdir,
        save_name + '.pkl'
    )

    sdf_dir = os.path.join(
        args.logdir,
        save_name + '_SDF'
    )

    # =========================================================
    # 5. Generation
    # =========================================================
    total_test = len(test_set)

    if args.limit is not None:
        total_test = min(
            args.limit,
            total_test
        )

    print()
    print('========================================')
    print('MolDiffdAE Template DDIM')
    print('========================================')
    print('Checkpoint :', args.ckpt)
    print('Noise      :', args.noise)
    print('Start step :', args.start_step)
    print('Test total :', total_test)
    print('PKL        :', pkl_path)
    print('SDF        :', sdf_dir)
    print('========================================')
    print()

    gen_dict = {}

    for n in range(total_test):

        print(
            f'[{n + 1}/{total_test}] generating...'
        )

        mol_list = sample_template_ddim(
            test_set[n],
            model,
            noise=args.noise,
            n_graphs=1,
            start_step=args.start_step
        )

        gen_dict[n] = mol_list

        print(
            '  valid molecules:',
            len(mol_list)
        )

    # =========================================================
    # 6. Save
    # =========================================================
    with open(pkl_path, 'wb') as f:
        pickle.dump(gen_dict, f)

    sdf_count = save_sdf(
        gen_dict,
        sdf_dir
    )

    print()
    print('========================================')
    print('Generation finished')
    print('========================================')
    print('PKL:', pkl_path)
    print('SDF:', sdf_dir)
    print('SDF count:', sdf_count)
    print('========================================')
