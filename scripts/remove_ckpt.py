import os

def delete_old_ckpt(ckpt_dir):
    checkpoint_list = os.listdir(ckpt_dir)
    # "epoch=xxx-step=xxx.ckpt"
    checkpoints = {}
    for cp in checkpoint_list:
        if cp.endswith('.ckpt'):
            epoch, step = cp.split('-')
            epoch = int(epoch.split('=')[1])
            step = int(step.split('=')[1].split('.')[0])
            if epoch not in checkpoints:
                checkpoints[epoch] = []
            checkpoints[epoch].append((step, cp))
    
    for epoch in checkpoints:
        # keep lastest 1 checkpoint
        checkpoints[epoch] = sorted(checkpoints[epoch], key = lambda x: x[0], reverse = True)[1:]
        for step, cp in checkpoints[epoch]:
            os.remove(os.path.join(ckpt_dir, cp))
            print(f'remove {cp}')
    
ckpt_dir = "logs/webvid_v8_2/version_7/checkpoints"
delete_old_ckpt(ckpt_dir)