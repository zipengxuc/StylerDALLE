import sys
import clip
import torchvision
from torch.utils.data import  DataLoader
from tqdm import tqdm
from ic_utils import *
from module import *
from dataset import StylerDALLERLDataset
from dall_e import map_pixels, unmap_pixels, load_model
from torch.distributions import Categorical
import torchvision.transforms as T


def update(model, flag=True):
    params = []
    for name, p in model.named_parameters():
        if "nat_dec" in name or "outNet" in name:
            print("update only", name)
            p.requires_grad = flag
            params.append(p)
    return params


def train(train_dataset, val_dataset, model, args, warmup_steps: int = 5000, exp_dir: str = "."):

    device = torch.device('cuda:0')
    dalle_dec = load_model('./dalle/decoder.pkl', device)
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
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=exp_dir)
        for idx, (caption, tokens_16, tokens_32, images_32, image_names) in enumerate(train_dataloader):
            model.zero_grad()
            bs = tokens_16.size(0)
            with torch.no_grad():
                dalle_encodings = dalle_dec.blocks.input(F.one_hot(tokens_16.to(device).view(bs, 16, 16), num_classes=8192).permute(0, 3, 1, 2).float())
            images_32 = images_32.to(device).squeeze()
            images_32 = torch.clamp((images_32 - 0.1) / (1 - 2 * 0.1), 0, 1)
            log_probs = model(dalle_encodings.permute(0, 2, 3, 1).view(bs, 256, 128))  # (bs, 256, 8192)
            # compute sampling log prob
            log_probs = log_probs.contiguous().view(-1, 8192)
            probs = torch.exp(log_probs)
            categorical_dist = Categorical(probs)
            sample = categorical_dist.sample()
            log_probs_new = categorical_dist.log_prob(sample).contiguous().view(bs, -1)
            z = F.one_hot(sample.contiguous().view(bs, 32, 32), num_classes=8192).permute(0, 3, 1, 2).float()
            with torch.no_grad():
                x_stats = dalle_dec(z).float()
                x_pred = torch.sigmoid(x_stats[:, :3])
                x_pred = torch.clamp((x_pred - 0.1) / (1 - 2 * 0.1), 0, 1)
                if idx % 500 == 0:
                    x_pred1 = T.ToPILImage(mode='RGB')(x_pred[0])
                    x_pred1 = T.ToTensor()(x_pred1)
                    x_pred2 = T.ToPILImage(mode='RGB')(images_32[0])
                    x_pred2 = T.ToTensor()(x_pred2)
                    torchvision.utils.save_image(torch.cat([x_pred1.unsqueeze(0), x_pred2.unsqueeze(0)]), './vis5/%s_%d_%d_train_pred.jpg' % (args.exp_dir, epoch, idx))
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
            progress.set_postfix({"loss": loss.item(), "reward1": mean_reward_batch.item(), "reward2": mean_reward_batch2.item()})
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
                    f.write('Train: Epoch - {}, CE - \n{}\n, base1 - \n{}\n, base2 - \n{}\n'.format(epoch, running_loss, baseline, baseline2))

                print(f">>> Validation epoch {epoch}")
                sys.stdout.flush()
                running_loss_v = 0.0
                baseline_v = 0.0
                for vidx, (caption, tokens_16, tokens_32, images_32, image_names) in enumerate(val_dataloader):
                    model.eval()
                    with torch.no_grad():
                        bs = tokens_16.size(0)
                        dalle_encodings = dalle_dec.blocks.input(F.one_hot(tokens_16.to(device).view(bs, 16, 16), num_classes=8192).permute(0, 3, 1, 2).float())
                        images_32 = images_32.to(device).squeeze()
                        images_32 = torch.clamp((images_32 - 0.1) / (1 - 2 * 0.1), 0, 1)
                        log_probs = model(dalle_encodings.permute(0, 2, 3, 1).view(bs, 256, 128))  # (bs, 256, 8192)
                        # compute sampling log prob
                        log_probs = log_probs.contiguous().view(-1, 8192)
                        probs = torch.exp(log_probs)
                        categorical_dist = Categorical(probs)
                        sample = categorical_dist.sample()
                        log_probs_new = categorical_dist.log_prob(sample).contiguous().view(bs, -1)
                        # compute reward: reconstructed image -> clip mse
                        z = F.one_hot(sample.view(bs, 32, 32), num_classes=8192).permute(0, 3, 1, 2).float()
                        x_stats = dalle_dec(z).float()
                        x_pred = torch.sigmoid(x_stats[:, :3])
                        x_pred = torch.clamp((x_pred - 0.1) / (1 - 2 * 0.1), 0, 1)
                        for pidx in range(bs):
                            x_pred1 = T.ToPILImage(mode='RGB')(x_pred[pidx])
                            x_pred1 = T.ToTensor()(x_pred1)
                            x_pred2 = T.ToPILImage(mode='RGB')(images_32[pidx])
                            x_pred2 = T.ToTensor()(x_pred2)
                            torchvision.utils.save_image(torch.cat([x_pred1.unsqueeze(0), x_pred2.unsqueeze(0)]), './vis5/%s_%d_%d_%d_val_pred.jpg' % (args.exp_dir, epoch, idx, pidx))
                        x_pred_r = T.Resize(224)(x_pred).to(dtype=torch.float16)
                        clip_pred = clip_model.encode_image(x_pred_r).to(dtype=torch.float32)  # (bs, 512)
                        clip_pred /= clip_pred.norm(dim=-1, keepdim=True)

                        text_styl = [args.styl + ' of ' + c for c in caption]

                        reward = torch.diag(clip_model(x_pred_r, clip.tokenize(text_styl).to(device))[0]).view(bs, 1) / 100
                        mean_reward_batch_v = torch.mean(reward)
                        rl_loss = -1 * log_probs_new * (reward.detach() - baseline_v)
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
    parser.add_argument('--train_data', default='./train')
    parser.add_argument('--val_data', default='./val')
    parser.add_argument('--work_dir', default='./ic')
    parser.add_argument('--exp_dir', default='oil_epm')
    parser.add_argument('--model_path', default='./sl3epm-024.pt')
    parser.add_argument('--styl', default='an oil painting of')
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--base', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--reward_coeff', type=float, default=1)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--res', type=int, default=32)
    parser.add_argument('--tok_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=4)
    args = parser.parse_args()
    model = StylerDALLEModel(args.tok_dim, args.hidden_dim, args.num_heads, args.num_layers, args.res)

    device = torch.device('cuda:0')

    with open('./train_cap.json', 'r') as f:
        train_data = json.load(f)
    with open('./val_cap.json', 'r') as f:
        val_data = json.load(f)
    train_dataset = StylerDALLERLDataset(train_data, args.train_data)
    val_dataset = StylerDALLERLDataset(val_data, args.val_data)

    sys.stdout.flush()
    train(train_dataset, val_dataset, model, args, exp_dir=args.exp_dir)
