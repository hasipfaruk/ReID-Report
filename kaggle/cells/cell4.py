# ===== CELL 4 : PAPER 1  TransReID (He et al., ICCV 2021) =====
# ViT-B/16 + overlapping patches (stride 12) + SIE (camera embedding) + JPM (jigsaw patch module).
# Settings follow the official configs/Market/vit_transreid_stride.yml.
try:
    import timm
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'timm'], check=False)
    import timm


def shuffle_unit(x, shift, groups):
    """JPM: shift the patch sequence (drop cls) then patch-shuffle it into `groups` interleaved groups."""
    B, _, D = x.shape
    x = torch.cat([x[:, shift:], x[:, 1:shift]], dim=1)
    if x.size(1) % groups:
        x = torch.cat([x, x[:, -2:-1].expand(B, groups - x.size(1) % groups, D)], dim=1)
    return x.view(B, groups, -1, D).transpose(1, 2).reshape(B, -1, D)


class TransReID(nn.Module):
    def __init__(self, num_classes, num_cams, img_size=(256, 128), stride=12, sie_coe=3.0,
                 jpm=True, shift=5, groups=2, divide=4, drop_path=0.1):
        super().__init__()
        try:   # jx_vit_base_p16_224 (ImageNet-21k pre-trained, 1k fine-tuned) = weights used by the paper
            base = timm.create_model('vit_base_patch16_224.orig_in21k_ft_in1k', pretrained=True, drop_path_rate=drop_path)
        except Exception as e:
            print('falling back to default vit_base_patch16_224 weights:', e)
            base = timm.create_model('vit_base_patch16_224', pretrained=True, drop_path_rate=drop_path)
        D = base.embed_dim
        self.grid = ((img_size[0] - 16) // stride + 1, (img_size[1] - 16) // stride + 1)   # 21 x 11 patches
        self.patch_proj = nn.Conv2d(3, D, 16, stride)                                       # overlapping patches
        self.patch_proj.load_state_dict(base.patch_embed.proj.state_dict())
        self.cls_token = base.cls_token
        pe = base.pos_embed.data                                                            # 1 x (1+14*14) x D
        g0 = int(math.sqrt(pe.shape[1] - 1))
        grid_pe = F.interpolate(pe[:, 1:].reshape(1, g0, g0, D).permute(0, 3, 1, 2), size=self.grid,
                                mode='bilinear', align_corners=False).permute(0, 2, 3, 1).reshape(1, -1, D)
        self.pos_embed = nn.Parameter(torch.cat([pe[:, :1], grid_pe], 1))
        self.sie_coe = sie_coe
        self.sie_embed = nn.Parameter(torch.zeros(num_cams, 1, D))
        nn.init.trunc_normal_(self.sie_embed, std=0.02)
        self.blocks = base.blocks[:-1]                                                      # first 11 blocks
        self.b1 = nn.Sequential(copy.deepcopy(base.blocks[-1]), copy.deepcopy(base.norm))   # global branch
        self.jpm, self.shift, self.groups, self.divide = jpm, shift, groups, divide
        if jpm:
            self.b2 = nn.Sequential(copy.deepcopy(base.blocks[-1]), copy.deepcopy(base.norm))  # shared local branch
        n = 1 + (divide if jpm else 0)
        self.bnnecks = nn.ModuleList([nn.BatchNorm1d(D) for _ in range(n)])
        self.classifiers = nn.ModuleList([nn.Linear(D, num_classes, bias=False) for _ in range(n)])
        for bn in self.bnnecks:
            nn.init.ones_(bn.weight); nn.init.zeros_(bn.bias); bn.bias.requires_grad_(False)
        for c in self.classifiers:
            nn.init.normal_(c.weight, std=0.001)

    def forward(self, x, cams):
        B = x.size(0)
        x = self.patch_proj(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], 1)
        x = x + self.pos_embed + self.sie_coe * self.sie_embed[cams]      # SIE: add camera embedding
        x = self.blocks(x)
        feats = [self.b1(x)[:, 0]]
        if self.jpm:
            L = (x.size(1) - 1) // self.divide
            tok, xs = x[:, :1], shuffle_unit(x, self.shift, self.groups)
            feats += [self.b2(torch.cat([tok, xs[:, i * L:(i + 1) * L]], 1))[:, 0] for i in range(self.divide)]
        if not self.training:      # NECK_FEAT 'before': global + local/4 features before BN
            return torch.cat([feats[0]] + [f / self.divide for f in feats[1:]], 1)
        logits = [c(bn(f)) for f, bn, c in zip(feats, self.bnnecks, self.classifiers)]
        return logits, feats


triplet_soft = rc.TripletLoss(margin=None)     # NO_MARGIN: True in the official config


def transreid_step(model, imgs, pids, cams, idx):
    logits, feats = model(imgs, cams)
    ce = [F.cross_entropy(l.float(), pids) for l in logits]          # IF_LABELSMOOTH: off
    tri = [triplet_soft(f, pids) for f in feats]
    if len(ce) == 1:
        return ce[0] + tri[0]
    # global branch weight 0.5, mean of the JPM local branches weight 0.5
    return 0.5 * (ce[0] + tri[0]) + 0.5 * (sum(ce[1:]) + sum(tri[1:])) / (len(ce) - 1)


TRANSREID_PAPER = {'market': {'standard': (95.2, 88.9)}}     # TransReID* (stride 12), no re-ranking


def run_transreid(dataset, name, fair=False):
    # 256x128, flip + pad 10/crop + random erasing 0.5, 16 ids x 4 imgs, mean/std 0.5,
    # SGD lr 0.008 (bias x2), wd 1e-4, 5 warm-up epochs + cosine, 120 epochs, no flip at test
    # fair=True: FAIR protocol (60 epochs, flip at test, last epoch reported); optimiser and losses unchanged.
    # mean/std 0.5 is kept: it is the normalisation the ViT weights were pre-trained with.
    if fair:
        data = rc.build_data(dataset, size=FAIR['size'], batch_size=FAIR['batch_size'], sampler='pk',
                             num_instances=FAIR['num_instances'], mean=[0.5] * 3, std=[0.5] * 3,
                             crop=FAIR['crop'], erasing=FAIR['erasing'], interp=FAIR['interp'])
    else:
        data = rc.build_data(dataset, size=(256, 128), batch_size=64, sampler='pk', num_instances=4,
                             mean=[0.5] * 3, std=[0.5] * 3, crop='pad', erasing=0.5, interp='bicubic')
    EPOCHS, BASE_LR = (2 if QUICK else (FAIR['epochs'] if fair else 120)), 0.008
    model = TransReID(data.num_classes, data.num_cams)
    groups = [{'params': [p], 'lr': BASE_LR * (2 if 'bias' in n else 1), 'weight_decay': 1e-4}
              for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.SGD(groups, lr=BASE_LR, momentum=0.9)
    sched = rc.warmup_cosine(opt, EPOCHS, warmup_epochs=5)
    model = model.to(rc.DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)                     # uses both T4 GPUs
    hist, best = rc.fit(model, data, opt, sched, transreid_step, EPOCHS, name,
                        eval_every=FAIR['eval_every'] if fair else 20, flip=fair and FAIR['flip_test'],
                        test_forward=lambda m, x, c: m(x, c), select_last=fair)
    if rc.STOPPED:
        return
    rc.plot_history(hist, name)
    rc.compare('TransReID' + (' (fair protocol)' if fair else ''), best, data,
               None if fair else TRANSREID_PAPER.get(dataset))
