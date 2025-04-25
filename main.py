import os, argparse, logging
from tqdm import tqdm
from libs.bounds import estimate_mutual_information
from libs.critics import set_critic, log_prob_gaussian
from libs.utils_gaussian import GaussianDataset
from libs.utils_images import ImageDataset, image_subset
from libs.utils_text import *
from libs.utils_mixture import *
from libs.models import mlp
from libs.encoder import *
from utils import *

# Initialize argument parser
parser = argparse.ArgumentParser()

# Define command-line arguments
parser.add_argument("--gpu_id", type=int, default=0)  # For CPU implementation, set -1
parser.add_argument("--savepath", type=str, default="results/gaussian")
parser.add_argument("--num-workers", type=int, default=64)

parser.add_argument("--ds", type=int, default=10, help="number of information sources")
parser.add_argument("--dr", type=int, default=10, help="representation dimension")

parser.add_argument("--dtype", type=str, default="text", choices=["gaussian", "image", "text", "mixture"]) 
parser.add_argument("--dname1", type=str, default="mnist", choices=["mnist", "cifar10", "cifar100"])  # Dataname for images
parser.add_argument("--dname2", type=str, default="imdb.bert-imdb-finetuned", choices=["imdb.bert-imdb-finetuned", "imdb.roberta-imdb-finetuned"])  # Dataname for texts
parser.add_argument("--nuisance", type=float, default=0)  # Only for images, [0, 1]

parser.add_argument("--output_scale", type=str, default="bit", choices=["bit", "nat"])

parser.add_argument("--critic_type", type=str, default="joint", choices=["joint", "separable", "bilinear", "inner"])
parser.add_argument("--critic_depth", type=str, default=2)
parser.add_argument("--critic_width", type=str, default=256)
parser.add_argument("--critic_embed", type=str, default=32)  # Only for separable critic
parser.add_argument("--estimator", type=str, default="smile-5", choices=["nwj", "js", "infonce", "dv", "mine", "smile-1", "smile-5", "smile-inf"]) 
# smile can have any threshold parameters, but we provide the examples of 1, 5, and inf for convenience.

parser.add_argument("--gaussian_cubic", type=int, default=0)
parser.add_argument("--image_patches", type=str, default="[1, 2, 5]")
parser.add_argument("--image_channels", type=int, default=1)
parser.add_argument("--encoder", type=str, default="None", choices=["None", "irevnet", "realnvp", "maf", "pretrained_resnet"])  # If use encoder function, image_channels should be 3
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=0.0005)
parser.add_argument("--n_steps", type=int, default=20000)
parser.add_argument("--mode", type=str, default="stepwise", choices=["stepwise", "single"]) 
# For stepwise, true MI values are set as [2, 4, 6, 8, 10]. For single, true MI value is set as defined in "true_mi" argument
parser.add_argument("--true_mi", type=float, default=2)  # Only for mode=single, should be in bit scale

args = parser.parse_args()

def main():
    # Set up logging
    logging.basicConfig()
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    
    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)
        
    file_handler = logging.FileHandler(os.path.join(args.savepath, "logs.log"))
    log.addHandler(file_handler)
    log.info(f'Logs will be saved at.. {args.savepath}')
    
    # Set GPU or CPU environment
    if args.gpu_id > 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
    log.info(device)
    
    # Define true mutual information values
    if args.mode == "stepwise":
        true_mi = [2, 4, 6, 8, 10]
    else:
        true_mi = args.true_mi
    log.info(true_mi)
    true_mi = bit2nat(true_mi, steps=args.n_steps)    
    
    image_patches = eval(args.image_patches)
    assert args.ds == np.prod(image_patches)
    assert np.max(true_mi) <= args.ds, "ds should be larger than the true MI"
    bsc_p = []
    for mi in true_mi:
        bsc_p.extend([cal_bsc(np.prod(image_patches), mi)])
    
        
    # Initialize encoder if specified
    if args.encoder != "None":
        if args.dtype == "image":
            assert args.image_channels == 3
            encoder_fn = eval(args.encoder) 
            encoder_fn = encoder_fn((args.image_channels, img_size, img_size)).to(device)
        elif args.dtype == "text":
            encoder_fn = eval(args.encoder)
            if args.encoder in ["irevnet", "pretrained_resnet"]:
                encoder_fn = encoder_fn((3, args.dr, 1)).to(device)
            else:
                encoder_fn = encoder_fn(args.dr).to(device)
    
    # Initialize datasets
    if args.dtype == "gaussian":
        dataset = GaussianDataset(true_mi, 
                                 args.ds, 
                                 args.batch_size, 
                                 cubic=args.gaussian_cubic)
    elif args.dtype in ["image", "text", "mixture"]:
        if args.dtype in ["image", "mixture"]:
            images, idx_dict = image_subset(args.dname1, subclass_list=[0, 1], grayscale=False)
            img_size = int(np.sqrt(args.dr // args.image_channels))
            image_dataset = ImageDataset(img_size, 
                                images, 
                                idx_dict, 
                                image_patches, 
                                bsc_p, 
                                args.batch_size, 
                                nuisance=args.nuisance)
        if args.dtype in ["text", "mixture"]:
            text_dataset = TextDataset(ds=args.ds, 
                                bsc_p=bsc_p, 
                                batch_size=args.batch_size, 
                                dataname=args.dname2, 
                                root="dataset")
            
        if args.dtype == "mixture":
            dataset = MixtureDataset(image_dataset, text_dataset)
        else:
            dataset = image_dataset if args.dtype == "image" else text_dataset

    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Define the function to train the estimator
    def train_estimator(critic_params, mi_params, opt_params, **kwargs):
        # Set critic based on encoder type
        if args.encoder in ["None", "maf", "realnvp"]:
            critic = set_critic(args.critic_type, args.dr, hidden_dim=args.critic_width, embed_dim=args.critic_embed, layers=args.critic_depth, device=device)
        elif args.encoder == "irevnet":
            critic = set_critic(args.critic_type, 3072*2*2, hidden_dim=args.critic_width, embed_dim=args.critic_embed, layers=args.critic_depth, device=device)
        elif args.encoder == "pretrained_resnet":
            critic = set_critic(args.critic_type, 2048, hidden_dim=args.critic_width, embed_dim=args.critic_embed, layers=args.critic_depth, device=device)
        
        baseline = None
        log.info(f'critic: {critic}')
        log.info(f'baseline: {baseline}')

        # Set optimizer
        opt_crit = opt_params['optimizer'](critic.parameters(), lr=opt_params['learning_rate'])
        if isinstance(baseline, torch.nn.Module):
            opt_base = opt_params['optimizer'](baseline.parameters(), lr=opt_params['learning_rate'])
        else:
            opt_base = None

        # Define training step
        def train_step(x, y, mi_params, buffer=None):
            opt_crit.zero_grad()
            if isinstance(baseline, torch.nn.Module):
                opt_base.zero_grad()
            
            if mi_params['estimator'] == "mine":
                mi, buffer = estimate_mutual_information(
                    mi_params['estimator'], x, y, critic, baseline_fn=baseline, buffer=buffer, **kwargs)
            else:
                mi = estimate_mutual_information(
                    mi_params['estimator'], x, y, critic, baseline_fn=baseline, buffer=buffer, **kwargs)
                buffer = None
            loss = -mi

            loss.backward(); opt_crit.step()
            if isinstance(baseline, torch.nn.Module):
                opt_base.step()

            return mi, buffer
        
        estimates = []
        buffer = None

        with tqdm(enumerate(dataloader), total=len(dataloader), desc="Training") as pbar:
            for i, batch in pbar:
                z1, z2 = batch
                
                with torch.no_grad():
                    if args.encoder == "irevnet":
                        _, z1 = encoder_fn(z1.to(device))
                        _, z2 = encoder_fn(z2.to(device))
                    elif args.encoder in ["realnvp", "maf"]:
                        z1, _ = encoder_fn(z1.view(args.batch_size, -1).to(device))
                        z2, _ = encoder_fn(z2.view(args.batch_size, -1).to(device))
                    elif args.encoder == "pretrained_resnet":
                        z1 = encoder_fn(z1.to(device))
                        z2 = encoder_fn(z2.to(device))
                
                mi, buffer = train_step(z1.view(args.batch_size, -1).cuda(), z2.view(args.batch_size, -1).cuda(), mi_params, buffer=buffer)
                
                mi = mi.detach().cpu().numpy()
                if args.output_scale == "bit":
                    mi = nat2bit(mi)
                    truth = nat2bit(true_mi[i])
                if (i+1) % 100 == 0:
                    pbar.set_description(f'Truth: {truth:.3f}, Average: {np.mean(estimates[i-100:i]):.3f}')
                
                estimates.append(mi)

            return np.array(estimates)

    # Set parameters for mutual information estimation
    mi_params = dict(estimator=args.estimator, critic=args.critic_type, baseline='unnormalized')
    critic_params = {
        'layers': args.critic_depth,
        'embed_dim': args.critic_embed,
        'hidden_dim': args.critic_width,
        'activation': 'relu',
        'normalization': None
    }

    opt_params = {
        'optimizer': torch.optim.Adam,
        'iterations': args.n_steps,
        'learning_rate': args.learning_rate,
    }

    # Train estimator and save results
    mis = train_estimator(critic_params, mi_params, opt_params)
    np.save(os.path.join(args.savepath, "mis.npy"), mis)

    print("======================================== END ESTIMATION ========================================")
    
    return mis

if __name__ == "__main__":
    print('======================================== START ESTIMATION =======================================')
    main()
