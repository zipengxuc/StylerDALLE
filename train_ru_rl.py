import sys
import clip
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
from ic_utils import *
from module import *
from dataset import StylerDALLERLDataset
from torch.distributions import Categorical
import torchvision.transforms as T
from rudalle import get_vae
from einops import rearrange


def update(model, flag=True):
    params = []
    for name, p in model.named_parameters():
        if "nat_dec" in name or "outNet" in name:
            print("update only", name)
            p.requires_grad = flag
            params.append(p)
    return params


def decode(vq, img_seq, shape=(32, 32)):
    img_seq = img_seq.view(img_seq.shape[0], -1)
    one_hot_indices = torch.nn.functional.one_hot(img_seq, num_classes=vq.num_tokens).float()
    z = (one_hot_indices @ vq.model.quantize.embed.weight)
    z = rearrange(z, 'b (h w) c -> b c h w', h=shape[0], w=shape[1])
    img = vq.model.decode(z)
    img = (img.clamp(-1., 1.) + 1) * 0.5
    return img


def train(train_dataset, val_dataset, model, args, warmup_steps: int = 5000, exp_dir: str = "."):

    device = torch.device('cuda:0')

    vqmodel = get_vae().to(device)
    vqmodel.eval().requires_grad_(False)

    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    batch_size = args.bs
    epochs = args.epochs
    lr = args.lr
    work_dir = args.work_dir
    output_dir = os.path.join(work_dir, exp_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    updating_params = update(model)
    optimizer = torch.optim.Adam(updating_params, lr=lr, betas=(0.9, 0.98), eps=1e-9)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.workers, drop_last=True)
    baseline = 0.0
    vis_path = '.'
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=exp_dir)
        for idx, (caption, _, tokens_16, tokens_32, images_32, image_names) in enumerate(train_dataloader):
            model.zero_grad()
            bs = tokens_16.size(0)
            with torch.no_grad():
                tokens_16 = tokens_16.view(bs, -1)
                one_hot_indices = torch.nn.functional.one_hot(tokens_16, num_classes=vqmodel.num_tokens).float().to(device)
                z_input = (one_hot_indices @ vqmodel.model.quantize.embed.weight)
            images_32 = images_32.to(device).squeeze()
            images_32 = torch.clamp((images_32 - 0.1) / (1 - 2 * 0.1), 0, 1)
            log_probs = model(z_input)  # (bs, 256, 8192)
            log_probs = log_probs.contiguous().view(-1, 8192)
            probs = torch.exp(log_probs)
            categorical_dist = Categorical(probs)
            sample = categorical_dist.sample()
            log_probs_new = categorical_dist.log_prob(sample).contiguous().view(bs, -1)
            with torch.no_grad():
                x_pred = decode(vqmodel, sample.contiguous().view(bs,-1)).float()
                if idx % 500 == 0:
                    x_pred1 = T.ToPILImage(mode='RGB')(x_pred[0])
                    x_pred1 = T.ToTensor()(x_pred1)
                    x_pred2 = T.ToPILImage(mode='RGB')(images_32[0])
                    x_pred2 = T.ToTensor()(x_pred2)
                    torchvision.utils.save_image(torch.cat([x_pred1.unsqueeze(0), x_pred2.unsqueeze(0)]), '%s/vis_ru/%s_%d_%d_train_pred.jpg' % (vis_path, args.exp_dir, epoch, idx))
                x_pred_r = T.Resize(224)(x_pred).to(dtype=torch.float16)
                clip_pred = clip_model.encode_image(x_pred_r).to(dtype=torch.float32)  # (bs, 512)
                clip_pred /= clip_pred.norm(dim=-1, keepdim=True)
                text_styl = [args.styl+' of '+c for c in caption]
                reward = torch.diag(clip_model(x_pred_r, clip.tokenize(text_styl).to(device))[0]).view(bs, 1) / 100  # clip similarity
                mean_reward_batch = torch.mean(reward)
            rl_loss = -1 * log_probs_new * (reward.detach() - baseline)
            loss = torch.mean(rl_loss) * args.reward_coeff
            loss.backward()
            _ = torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item(), "reward1": mean_reward_batch.item()})
            progress.update()
            running_loss = bs / (bs + 1) * running_loss + 1 / (bs + 1) * loss.data
            baseline = bs / (bs + 1) * baseline + 1 / (bs + 1) * mean_reward_batch.data
            if (idx+1) % 5000 == 0:
                cid = (idx +1) / 5000
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{exp_dir}_{epoch:03d}_{int(cid):02d}.pt"),
                )
                with open(os.path.join(output_dir, 'logs.txt'), 'a') as f:
                    f.write('Train: Epoch - {}, CE - \n{}\n, base1 - \n{}\n'.format(epoch, running_loss, baseline))

                print(f">>> Validation epoch {epoch}")
                sys.stdout.flush()
                running_loss_v = 0.0
                baseline_v = 0.0
                for vidx, (caption, _, tokens_16, tokens_32, images_32, image_names) in enumerate(val_dataloader):
                    model.eval()
                    with torch.no_grad():
                        bs = tokens_16.size(0)
                        tokens_16 = tokens_16.view(bs, -1)
                        one_hot_indices = torch.nn.functional.one_hot(tokens_16,
                                                                      num_classes=vqmodel.num_tokens).float().to(device)
                        z_input = (one_hot_indices @ vqmodel.model.quantize.embed.weight)

                        images_32 = images_32.to(device).squeeze()
                        images_32 = torch.clamp((images_32 - 0.1) / (1 - 2 * 0.1), 0, 1)
                        log_probs = model(z_input)  # (bs, 256, 8192)
                        log_probs = log_probs.contiguous().view(-1, 8192)
                        probs = torch.exp(log_probs)
                        categorical_dist = Categorical(probs)
                        sample = categorical_dist.sample()
                        log_probs_new = categorical_dist.log_prob(sample).contiguous().view(bs, -1)
                        x_pred = decode(vqmodel, sample.contiguous().view(bs,-1)).float()
                        for pidx in range(bs):
                            x_pred1 = T.ToPILImage(mode='RGB')(x_pred[pidx])
                            x_pred1 = T.ToTensor()(x_pred1)
                            x_pred2 = T.ToPILImage(mode='RGB')(images_32[pidx])
                            x_pred2 = T.ToTensor()(x_pred2)
                            torchvision.utils.save_image(torch.cat([x_pred1.unsqueeze(0), x_pred2.unsqueeze(0)]), '%s/vis_ru/%s_%d_%d_%d_val_pred.jpg' % (vis_path, args.exp_dir, epoch, idx, pidx))
                        x_pred_r = T.Resize(224)(x_pred).to(dtype=torch.float16)
                        clip_pred = clip_model.encode_image(x_pred_r).to(dtype=torch.float32)  # (bs, 512)
                        clip_pred /= clip_pred.norm(dim=-1, keepdim=True)
                        text_styl = [args.styl + ' of ' + c for c in caption]
                        reward = torch.diag(clip_model(x_pred_r, clip.tokenize(text_styl).to(device))[0]).view(bs, 1) / 100
                        mean_reward_batch_v = torch.mean(reward)
                        rl_loss = -1 * log_probs_new * (reward.detach() - baseline)
                        val_loss = torch.mean(rl_loss) * args.reward_coeff
                        running_loss_v = bs / (bs + 1) * running_loss_v + 1 / (bs + 1) * val_loss.data
                        baseline_v = bs / (bs + 1) * baseline_v + 1 / (bs + 1) * mean_reward_batch_v.data
                        if idx > 20*bs:
                            with open(os.path.join(output_dir, 'logs.txt'), 'a') as f:
                                f.write('Val: Epoch - {}, CE - \n{}\n, base1 - \n{}\n'.format(epoch, running_loss_v,
                                                                                                                baseline_v))
                            break
                progress.close()

                if epoch % args.save_every == 0 or epoch == epochs - 1:
                    torch.save(
                        model.state_dict(),
                        os.path.join(output_dir, f"{exp_dir}-{epoch:03d}.pt"),
                    )

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', default='./train_ru')
    parser.add_argument('--val_data', default='./val_ru')
    parser.add_argument('--work_dir', default='./ic')
    parser.add_argument('--exp_dir', default='vg_ruc')
    parser.add_argument('--model_path', default='./sl_ru-015.pt')
    parser.add_argument('--styl', default='a Van Gogh style oil painting of')
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--base', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--reward_coeff', type=float, default=1)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--res', type=int, default=32)
    parser.add_argument('--tok_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=4)
    args = parser.parse_args()
    model = StylerDALLEModel(args.tok_dim, args.hidden_dim, args.num_heads, args.num_layers, args.res)
    device = torch.device('cuda:0')

    vis_path = '.'

    with open('%s/train_cap.json' % vis_path, 'r') as f:
        train_data = json.load(f)
    with open('%s/val_cap.json' % vis_path, 'r') as f:
        val_data = json.load(f)
    train_dataset = StylerDALLERLDataset(train_data, args.train_data)
    val_dataset = StylerDALLERLDataset(val_data, args.val_data)

    sys.stdout.flush()
    train(train_dataset, val_dataset, model, args, exp_dir=args.exp_dir)
