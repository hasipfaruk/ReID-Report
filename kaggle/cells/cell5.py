# ===== CELL 5 : PAPER 2  CAL, Clothes-based Adversarial Loss (Gu et al., CVPR 2022) =====
# Settings follow the official Simple-CCReID code (configs/default_img.py + res50_cels_cal.yaml).
class CALNet(nn.Module):
    """ResNet-50 (last stride 1) -> concat(global max pool, global avg pool) = 4096-d -> BN.
    classifier: identity classifier (linear); clothes_classifier: cosine classifier over all training outfits."""
    def __init__(self, num_pids, num_clothes, dim=4096):
        super().__init__()
        r = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        r.layer4[0].conv2.stride = (1, 1)
        r.layer4[0].downsample[0].stride = (1, 1)
        self.base = nn.Sequential(*list(r.children())[:-2])
        self.bn = nn.BatchNorm1d(dim)
        nn.init.normal_(self.bn.weight, 1.0, 0.02); nn.init.zeros_(self.bn.bias)
        self.classifier = nn.Linear(dim, num_pids)
        nn.init.normal_(self.classifier.weight, std=0.001); nn.init.zeros_(self.classifier.bias)
        self.clothes_weight = nn.Parameter(torch.empty(num_clothes, dim).uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5))

    def forward(self, x):
        x = self.base(x)
        return self.bn(torch.cat([F.adaptive_max_pool2d(x, 1), F.adaptive_avg_pool2d(x, 1)], 1).flatten(1))

    def clothes_cos(self, f, detach_weight=False):
        w = self.clothes_weight.detach() if detach_weight else self.clothes_weight
        return F.normalize(f, dim=1) @ F.normalize(w, dim=1).t()


def clothes_adversarial_loss(cos, targets, pos_mask, scale=16.0, epsilon=0.1):
    """Eq. of the CAL paper: pull the feature towards ALL outfits of its own identity (positives), push away from
    outfits of other identities, so the backbone cannot tell outfits of the same person apart."""
    x = scale * cos
    neg_mask = 1 - pos_mask
    id_mask = F.one_hot(targets, x.size(1)).float()
    exp = torch.exp(x)
    log_prob = x - torch.log((exp * neg_mask).sum(1, keepdim=True) + exp)
    mask = (1 - epsilon) * id_mask + epsilon / pos_mask.sum(1, keepdim=True) * pos_mask
    return (-mask * log_prob).sum(1).mean()


CAL_STATE = {}


def cal_step(model, imgs, pids, cams, idx):
    st = CAL_STATE
    f = model(imgs)
    loss = F.cross_entropy(model.classifier(f).float(), pids, label_smoothing=0.1)    # identity loss (CE + LS)
    if st['adversarial'] and st['epoch'] >= st['start_cc']:
        clothes = st['clothes'][idx].to(rc.DEVICE)
        with torch.autocast('cuda', enabled=False):
            f32 = f.float()
            # (1) clothes classifier: CosFace(s=16, m=0) on DETACHED features -> only the classifier learns
            loss = loss + F.cross_entropy(16.0 * model.clothes_cos(f32.detach()), clothes)
            # (2) adversarial loss on the backbone, classifier weights frozen for this term
            if st['epoch'] >= st['start_adv']:
                loss = loss + clothes_adversarial_loss(model.clothes_cos(f32, detach_weight=True),
                                                       clothes, st['pid2clothes'][pids])
    return loss


CAL_PAPER = {'prcc': {'same_clothes': (100.0, 99.8), 'clothes_changing': (55.2, 55.8)},
             'ltcc': {'general': (74.2, 40.8), 'clothes_changing': (40.1, 18.0)}}


def run_cal(dataset, adversarial, name, fair=False):
    # 384x192, random crop 0.5 + flip 0.5 + random erasing 0.5, batch 32 (8 imgs per id),
    # Adam lr 3.5e-4 wd 5e-4, lr x0.1 at epochs 20 and 40 (clothes classifier keeps a constant lr),
    # 60 epochs, clothes classifier + adversarial loss from epoch 25, flip at test, eval every 5 epochs
    # fair=True: FAIR protocol (input size, batch, augmentation, last epoch reported); optimiser and losses unchanged.
    if fair:
        data = rc.build_data(dataset, size=FAIR['size'], batch_size=FAIR['batch_size'], sampler='pk',
                             num_instances=FAIR['num_instances'], crop=FAIR['crop'], erasing=FAIR['erasing'],
                             interp=FAIR['interp'])
    else:
        data = rc.build_data(dataset, size=(384, 192), batch_size=32, sampler='pk', num_instances=8,
                             crop='cal', erasing=0.5)
    EPOCHS, START = (2, 1) if QUICK else ((FAIR['epochs'], 25) if fair else (60, 25))
    model = CALNet(data.num_classes, data.num_clothes)
    CAL_STATE.update(adversarial=adversarial, start_cc=START, start_adv=START, epoch=0,
                     clothes=data.train_clothes, pid2clothes=data.pid2clothes.to(rc.DEVICE))
    backbone = [p for n, p in model.named_parameters() if n != 'clothes_weight']
    opt = torch.optim.Adam([{'params': backbone}, {'params': [model.clothes_weight]}], lr=3.5e-4, weight_decay=5e-4)
    step_decay = lambda e: 0.1 ** sum(e >= m for m in (20, 40))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, [step_decay, lambda e: 1.0])
    hist, best = rc.fit(model, data, opt, sched, cal_step, EPOCHS, name,
                        eval_every=FAIR['eval_every'] if fair else 5, flip=True,
                        start_hook=lambda m, ep: CAL_STATE.update(epoch=ep), select_last=fair)
    if rc.STOPPED:
        return
    rc.plot_history(hist, name)
    rc.compare(('CAL' if adversarial else 'Baseline (no CAL)') + (' (fair protocol)' if fair else ''), best, data,
               CAL_PAPER.get(dataset) if adversarial and not fair else None)
