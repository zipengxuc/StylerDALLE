import sys
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from ic_utils import *
from module import *
from dataset import StylerDALLESLDataset
from dall_e import map_pixels, unmap_pixels, load_model


def train(train_dataset, val_dataset, model, args, warmup_steps: int = 5000, exp_dir: str = "."):

    device = torch.device('cuda:0')
    dalle_dec = load_model('dalle/decoder.pkl', device)
    batch_size = args.bs
    epochs = args.epochs
    lr = args.lr
    work_dir = args.work_dir
    output_dir = os.path.join(work_dir, exp_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.workers, drop_last=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=exp_dir)
        for idx, (tokens_16, tokens_32, image_names) in enumerate(train_dataloader):
            model.zero_grad()
            bs = tokens_16.size(0)
            with torch.no_grad():
                dalle_encodings = dalle_dec.blocks.input(F.one_hot(tokens_16.to(device).view(bs, 16, 16), num_classes=8192).permute(0, 3, 1, 2).float())
            tokens = tokens_32.to(device)
            outputs = model(dalle_encodings.permute(0, 2, 3, 1).view(bs, 16*16, 128))
            loss = maskedNll(outputs, tokens.contiguous().view(bs, -1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
            running_loss = args.bs / (args.bs + 1) * running_loss + 1 / (args.bs + 1) * loss.data

        torch.save(
            model.state_dict(),
            os.path.join(output_dir, f"{exp_dir}_latest.pt"),
        )
        with open(os.path.join(output_dir, 'logs.txt'), 'a') as f:
            f.write('Train: Epoch - {}, CE - \n{}\n'.format(epoch, running_loss))
        progress.close()

        print(f">>> Validation epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(val_dataloader), desc=exp_dir)
        running_loss = 0.0
        for idx, (tokens_16, tokens_32, image_names) in enumerate(val_dataloader):
            model.eval()
            with torch.no_grad():
                bs = tokens_16.size(0)
                dalle_encodings = dalle_dec.blocks.input(F.one_hot(tokens_16.to(device).view(bs, 16, 16), num_classes=8192).permute(0, 3, 1, 2).float())
                tokens = tokens_32.to(device)
                outputs = model(dalle_encodings.permute(0, 2, 3, 1).view(bs, 256, 128))
                val_loss = maskedNll(outputs, tokens.contiguous().view(bs, -1))
                progress.set_postfix({"loss": val_loss.item()})
                progress.update()
                running_loss = args.bs / (args.bs + 1) * running_loss + 1 / (args.bs + 1) * val_loss.data
        with open(os.path.join(output_dir, 'logs.txt'), 'a') as f:
            f.write('Val: Epoch - {}, \n{}\n'.format(epoch, running_loss))

        progress.close()

        if epoch % args.save_every == 0 or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{exp_dir}-{epoch:03d}.pt"),
            )

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='./prep_coco/train')
    parser.add_argument('--val_path', default='./prep_coco/val')
    parser.add_argument('--work_dir', default='./ic')
    parser.add_argument('--exp_dir', default='sl')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--res', type=int, default=32)
    parser.add_argument('--tok_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=4)
    args = parser.parse_args()
    model = StylerDALLEModel(args.tok_dim, args.hidden_dim, args.num_heads, args.num_layers, args.res)
    device = torch.device('cuda:0')
    train_dataset = StylerDALLESLDataset(args.train_path)
    val_dataset = StylerDALLESLDataset(args.val_path)
    sys.stdout.flush()
    train(train_dataset, val_dataset, model, args, exp_dir=args.exp_dir)
