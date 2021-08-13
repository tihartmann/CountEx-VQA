import sys
sys.path.append("./")
import torch
import torch.nn
import config
from torchvision.utils import save_image

def save_some_examples(gen, val_loader, epoch, folder):
    for idx, (img, mask, logits, q_emb, qid) in enumerate(val_loader):
        if idx == 2:
            break
        img, mask, logits, q_emb = img.to(config.DEVICE), mask.to(config.DEVICE), logits.to(config.DEVICE), q_emb.to(config.DEVICE)
        qid = qid.item()
        gen.eval()
        with torch.no_grad():
            y_fake = gen(img, q_emb, logits, mask)
            y_fake = y_fake
            save_image(y_fake, folder + f"/y_gen_qid_{qid}_{epoch}.png")
            if epoch == 0:
                save_image(img, folder + f"/input_qid_{qid}_{epoch}.png")
            if epoch == 1:
                save_image(img, folder + f"/label_qid_{qid}_{epoch}.png")
        gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
        
def gradient_penalty(critic, real, fake, logits_new, logits_old, device=config.DEVICE):
    BATCH_SIZE, C, H, W = real.shape
    _, ans_shape = logits_new.shape
    alpha = torch.rand((BATCH_SIZE, 1))
    alpha_img = torch.tensor([[[[torch.squeeze(alpha)]]]]).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha_img + fake * (1 - alpha_img)
    alpha_logits = alpha.repeat(1,ans_shape).to(device)
    interpolated_logits = logits_old * alpha_logits + logits_new * (1 - alpha_logits)
    # Calculate critic scores
    mixed_scores = critic(interpolated_images, interpolated_logits)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

