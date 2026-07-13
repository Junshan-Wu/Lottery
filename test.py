import torch
import numpy as np
import torch.nn.functional as F
import DataLoader
import parameters


def test(model):
    print("Start testing")
    model.eval()
    testset = DataLoader.Test_data_Loader()
    params = parameters.get_parameters()
    device = params.device if torch.cuda.is_available() else "cpu"
    test_correct = 0
    test_num_all = 0
    total_loss = 0
    with torch.no_grad():
        for step, (images, labels) in enumerate(testset):
            images = images.to(device)
            labels = labels.to(device)
            batch_size = labels.size(0)
            preds = model(images) # 得到batchsize*clas(10)的张量
            loss = F.cross_entropy(preds, labels)
            total_loss += loss.item() * batch_size
            probs, label_preds = preds.max(dim=1)
            correct_preds = torch.sum(labels==label_preds)
            batch_acc = correct_preds.item() / batch_size
            test_correct += correct_preds
            test_num_all += batch_size
            if step % 10 == 0:
                print(f"step: {step} batch success rate: {batch_acc:.3f}, loss: {loss.item():.3f}")

    avg_loss = total_loss / test_num_all
    success_rate = (test_correct/test_num_all).item()
    print(f"Finish! The final success rate is {success_rate:.3f}, and the average loss is {avg_loss:.3f}")

    return success_rate, avg_loss




