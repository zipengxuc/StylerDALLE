import sys
import clip
from torch.utils.data import DataLoader
from ic_utils import *
from module import *
from dataset import StylerDALLERLDataset
from tqdm import tqdm
import torchvision.transforms as T
from torch.distributions import Categorical
from dall_e import map_pixels, unmap_pixels, load_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', default='./train')
    parser.add_argument('--val_data', default='./val')
    parser.add_argument('--work_dir', default='./ic')
    parser.add_argument('--model_path', default='./ckpts/model.pt')
    parser.add_argument('--stylp', default='a water color painting')
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--res', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')

    args = parser.parse_args()
    device = torch.device('cuda:0')
    model = StylerDALLEModel(args.tok_dim, args.hidden_dim, args.num_heads, args.num_layers, args.res)

    with open('./val_cap.json', 'r') as f:
        val_data = json.load(f)

    val_dataset = StylerDALLERLDataset(val_data, args.val_data)
    val_dataloader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, drop_last=True)

    model_path = args.model_path
    if os.path.isfile(model_path):
        print(f"loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"{model_path} is not exist")

    enc = load_model("./dalle/encoder.pkl", device)
    dec = load_model("./dalle/decoder.pkl", device)
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)

    print(f">>> NamesRead")
    print(f">>> Validation")
    sys.stdout.flush()
    sp_score = 0.0
    counter = 0.0
    model.to(device)
    progress = tqdm(total=len(val_dataloader), desc=args.work_dir)
    for idx, (caption, tokens_16, tokens_32, images_32, image_names) in enumerate(val_dataloader):
        model.eval()
        bs = tokens_16.size(0)
        with torch.no_grad():
            dalle_encodings = dec.blocks.input(
                F.one_hot(tokens_16.to(device).view(bs, 16, 16), num_classes=8192).permute(0, 3, 1, 2).float())

            '''stylized image'''
            outputs = model(dalle_encodings.permute(0, 2, 3, 1).view(bs, 256, 128))
            log_probs = outputs.contiguous().view(-1, 8192)
            probs = torch.exp(log_probs)
            categorical_dist = Categorical(probs)
            sample = categorical_dist.sample()
            log_probs_new = categorical_dist.log_prob(sample).contiguous().view(bs, -1)
            z = F.one_hot(sample.view(bs, 32, 32), num_classes=8192).permute(0, 3, 1, 2).float()
            x_stats = dec(z).float()
            x_pred = torch.sigmoid(x_stats[:, :3])
            x_pred = torch.clamp((x_pred - 0.1) / (1 - 2 * 0.1), 0, 1)

            x_pred_r = T.Resize(224)(x_pred).to(dtype=torch.float16)
            clip_pred = clip_model.encode_image(x_pred_r).to(dtype=torch.float32)  # (bs, 512)
            clip_pred /= clip_pred.norm(dim=-1, keepdim=True)
            text_stylp = [args.stylp for c in caption]

            sim_pp = torch.mean(torch.diag(clip_model(x_pred_r, clip.tokenize(text_stylp).to(device))[0]).to(dtype=torch.float32))
            counter += 1
            sp_score += sim_pp
            progress.update()

        if idx % 500 == 0:
            with open('./eval_results/%s_log.txt' % args.stylp, 'a') as f:
                f.write('Val: idx - {}, sp - {}'.format(idx, sp_score/counter))

    with open('./eval_results/%s_log.txt' % args.stylp, 'a') as f:
        f.write('Val-Total: idx - {},sp - {}'.format(idx, sp_score/counter))

    sp_score = sp_score/counter
    print(sp_score)
    progress.close()
