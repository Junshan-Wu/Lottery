import torch
import torch.nn.functional as F
import DataLoader
import os
import parameters
import torch.optim.lr_scheduler as lr_scheduler


def train_full(model, num_epoch):
    print("Start training (full training set, no validation)")
    model.train()
    trainset = DataLoader.Full_train_data_Loader()
    params = parameters.get_parameters()
    device = params.device if torch.cuda.is_available() else "cpu"
    base_lr = params.learning_rate
    warmup_epochs = params.warmup_epochs
    warmup_enabled = warmup_epochs != 0

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    effective_epochs = num_epoch - warmup_epochs if warmup_enabled else num_epoch
    if params.lr_scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=effective_epochs)
    elif params.lr_scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    elif params.lr_scheduler == 'exponential':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        scheduler = None  # No scheduler

    loss_history = []  # 记录每个 epoch 的平均训练损失
    success_history = []  # 记录每个 epoch 的训练成功率
    train_step_losses = []  # 记录每个训练step的损失值
    saveDir = params.model_save_dir
    os.makedirs(saveDir, exist_ok=True)

    global_step = 0  # 全局训练步数

    best_train_loss = float('inf')
    best_model_path = None

    def _name_safe(value):
        return str(value).replace('.', 'p').replace('-', 'm')

    for epoch in range(num_epoch):
        if warmup_enabled and epoch < warmup_epochs:
            warmup_lr = base_lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr

        correct_preds_all = 0
        train_num_all = 0
        total_loss = 0

        model.train()
        for step, (images, labels) in enumerate(trainset):
            global_step += 1
            images = images.to(device)
            labels = labels.to(device)
            batch_size = labels.size(0)
            optimizer.zero_grad()
            preds = model(images)
            loss = F.cross_entropy(preds, labels)
            total_loss += loss.item() * batch_size
            train_step_losses.append(loss.item())
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                _, label_preds = preds.max(dim=1)
                correct_preds = torch.sum(labels == label_preds)
                success_rate = correct_preds / batch_size
                correct_preds_all += correct_preds
                train_num_all += batch_size
                if step % 10 == 0:
                    print(f'epoch: {epoch} train step: {step} accuracy: {success_rate:.3f}, loss: {loss.item():.3f}')

        # Update learning rate after epoch.
        # For cosine with warm-up, use shifted epoch index so decay starts right after warm-up.
        if scheduler:
            if warmup_enabled:
                if epoch >= warmup_epochs:
                    scheduler.step(epoch - warmup_epochs + 1)
            else:
                scheduler.step()

        avg_loss = total_loss / train_num_all
        success_rate_epoch = (correct_preds_all / train_num_all).item()
        print('epoch {} training finish! The accuracy is {}, and the average loss is {}'.format(epoch, success_rate_epoch, avg_loss))

        success_history.append(success_rate_epoch)
        loss_history.append(avg_loss)

        if avg_loss < best_train_loss:
            best_train_loss = avg_loss
            best_model_name = (
                f"fulltrain_lr_{_name_safe(params.learning_rate)}_"
                f"bs_{params.batch_size}_"
                f"sch_{params.lr_scheduler}_"
                f"act_{params.activation}_"
                f"cutout_{params.use_cutout}_"
                f"warmup_{params.warmup_epochs}_"
                f"valid_{_name_safe(params.valid_size)}.pth"
            )
            best_model_path = os.path.join(
                saveDir,
                best_model_name
            )
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model updated and saved to {best_model_path} with training loss {best_train_loss:.4f}")

    if best_model_path:
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model weights from {best_model_path}")

    return model, loss_history, success_history, train_step_losses
