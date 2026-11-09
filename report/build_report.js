// Builds the ReID report (Word). Plain styling: black text only, grey-scale figures, no em or en dashes.
const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, BorderStyle, ImageRun, TableOfContents, Footer, PageNumber, LevelFormat,
} = require('docx');

const FIG = path.join(__dirname, 'figures');
const OUT = path.join(__dirname, 'ReID_Report_Fair_Comparison_PRCC.docx');
const BLACK = '000000';
const FONT = 'Times New Roman';
const PAGE_W = 9026; // A4 width minus 2 x 1440 margins (DXA)

// ------------------------------------------------------------------ helpers
// inline markup: **bold**, *italic*
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith('**')) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
const P = (text, opts = {}) => new Paragraph({ children: runs(text), spacing: { after: 120, line: 288 }, alignment: AlignmentType.JUSTIFIED, ...opts });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)], pageBreakBefore: true });
const H1n = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const B = (text, level = 0) => new Paragraph({ children: runs(text), numbering: { reference: 'bullets', level }, spacing: { after: 60, line: 276 } });
const N = (text, ref = 'numbers') => new Paragraph({ children: runs(text), numbering: { reference: ref, level: 0 }, spacing: { after: 60, line: 276 } });
const EQ = (text) => new Paragraph({ children: [new TextRun({ text, italics: true })], alignment: AlignmentType.CENTER, spacing: { before: 80, after: 120 } });
const CAP = (text) => new Paragraph({ children: runs(text, { size: 20 }), alignment: AlignmentType.CENTER, spacing: { before: 60, after: 240 } });

let figNo = 0, tabNo = 0;
function figure(file, caption, widthIn = 6.2) {
  figNo += 1;
  const buf = fs.readFileSync(path.join(FIG, file));
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20); // PNG header
  const width = Math.round(widthIn * 96), height = Math.round(width * h / w);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 }, keepNext: true,
      children: [new ImageRun({ type: 'png', data: buf, transformation: { width, height } })] }),
    CAP(`**Figure ${figNo}.** ${caption}`),
  ];
}
const border = { style: BorderStyle.SINGLE, size: 4, color: BLACK };
const borders = { top: border, bottom: border, left: border, right: border };
function table(caption, header, rows, widths, opts = {}) {
  tabNo += 1;
  const total = widths.reduce((a, b) => a + b, 0);
  const mk = (cells, bold) => new TableRow({
    tableHeader: bold, cantSplit: true,
    children: cells.map((c, i) => new TableCell({
      borders, width: { size: widths[i], type: WidthType.DXA },
      margins: { top: 50, bottom: 50, left: 90, right: 90 },
      children: [new Paragraph({ alignment: i === 0 && !opts.centerFirst ? AlignmentType.LEFT : AlignmentType.CENTER,
        children: runs(String(c), { size: 19, bold: bold || undefined }) })],
    })),
  });
  return [
    new Paragraph({ children: runs(`**Table ${tabNo}.** ${caption}`, { size: 20 }), alignment: AlignmentType.CENTER, spacing: { before: 200, after: 80 }, keepNext: true }),
    new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, alignment: AlignmentType.CENTER,
      rows: [mk(header, true), ...rows.map(r => mk(r, false))] }),
    new Paragraph({ children: [], spacing: { after: 120 } }),
  ];
}
const even = (n, total = PAGE_W) => Array(n).fill(Math.floor(total / n));

// ------------------------------------------------------------------ content
const body = [];
const add = (...xs) => xs.forEach(x => Array.isArray(x) ? body.push(...x) : body.push(x));

// ---------- title page
add(
  new Paragraph({ children: [], spacing: { before: 2200 } }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
    children: [new TextRun({ text: 'Clothes-Changing Person Re-Identification on PRCC', bold: true, size: 40 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 480 },
    children: [new TextRun({ text: 'A Comparison of a ResNet-50 Baseline, TransReID and CAL Under One Training Protocol', bold: true, size: 30 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 1600 },
    children: [new TextRun({ text: 'Research report', size: 26 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: 'Prepared by: [Your name]', size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: 'Submitted to: [Supervisor name]', size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: 'November 2026', size: 24 })] }),
);

// ---------- abstract + contents
add(
  H1('Abstract'),
  P('In clothes-changing person re-identification, a system has to recognise a person in a new camera even when that person is wearing different clothes. In an earlier stage of this project we reproduced two published methods, TransReID (ICCV 2021) and CAL (CVPR 2022), each with the settings from its own paper. Those numbers cannot be compared directly, because each method was trained with a different image size, batch, number of epochs and test procedure. In this report we compare three models on one task, clothes-changing re-identification on the PRCC dataset, and retrain all of them under a single protocol. The models are a plain ResNet-50 baseline, TransReID (ViT-B/16) and CAL (ResNet-50 with the clothes-based adversarial loss).'),
  P('The shared protocol fixes everything that is not part of a method: the training data, a 256 x 128 input, a batch of 16 identities with 4 images each, the same augmentation, 60 epochs, flip at test time, and reporting the last epoch instead of the best one. Each model keeps its own network, losses and optimiser.'),
  P('With this protocol the plain baseline gave the best final result, **51.03% Rank-1 and 51.67% mAP** in the clothes-changing setting. TransReID reached 45.67% and 48.28%, and CAL reached 46.32% and 48.04%. CAL was the best model at epoch 30, five epochs after its adversarial loss was switched on (51.45% Rank-1 and mAP), but it lost about 3.4 points of mAP over the remaining 30 epochs. Under each paper\'s own settings, CAL was clearly the best (54.23% mAP at epoch 60, against 45.56% for the baseline). The advantage of CAL therefore depends strongly on how it is trained. These results come from one training run per model, and two more runs per model are planned to measure the run-to-run variation.'),
  H1('Contents'),
  new TableOfContents('Contents', { hyperlink: true, headingStyleRange: '1-2' }),
);

// ---------- 1 introduction
add(
  H1('1 Introduction'),
  H2('1.1 The task'),
  P('Person re-identification (ReID) takes an image of a person, called the *query*, and ranks a large set of images from other cameras, called the *gallery*, so that images of the same person come first. It is used in multi-camera tracking and video search. Most ReID benchmarks were recorded over a short time, so everyone wears the same clothes in every camera. Models trained on them learn to rely on clothing colour and texture, because that is the easiest way to tell people apart.'),
  P('In real long-term use, people change their clothes. A model that matches people by their clothes then fails. Clothes-changing ReID asks the model to use cues that stay the same when the outfit changes, such as body shape, hair, face and the way a person stands or walks. PRCC is a standard benchmark for this task: each person is filmed in one outfit by two cameras and in a different outfit by a third camera.'),
  H2('1.2 Why a common protocol is needed'),
  P('Our earlier reproduction trained TransReID on Market-1501 and CAL on PRCC, each with its official settings. Both reproductions matched their papers closely (Section 6.1). However, two results obtained on different datasets, or on the same dataset with different training settings, do not tell us which method is better. CAL, for example, was trained with larger images (384 x 192) than TransReID (256 x 128), and TransReID was trained for twice as many epochs. Any difference between them could come from these settings instead of from the methods themselves.'),
  P('The goal of this report is therefore narrower and more controlled: one task, one dataset, one training protocol, and three models trained and tested again from scratch.'),
  H2('1.3 What we did'),
  N('Fixed the task as clothes-changing ReID on PRCC, with the standard PRCC test split.'),
  N('Chose three models: a ResNet-50 baseline, TransReID and CAL. The baseline is the CAL network without the clothes losses, so the CAL loss is the only difference between those two.'),
  N('Defined one protocol for everything that is not part of a method (Section 5.2), and retrained all three models with it on the same hardware.'),
  N('Compared the results with each other and with the results of the same models under their original settings.'),
  H2('1.4 Main findings'),
  B('Under the common protocol, the baseline had the best final clothes-changing result (51.03% Rank-1, 51.67% mAP). TransReID and CAL ended about 3.5 points of mAP lower and very close to each other.'),
  B('CAL helped for a short time. Right after its adversarial loss was switched on, it moved from below the baseline to above it, but this lead did not last until the end of training.'),
  B('The training protocol changed the ranking. With its own settings CAL beat the baseline by 8.67 points of mAP. With the common settings it ended 3.63 points below it.'),
  B('All three models were almost perfect when clothes did not change (Rank-1 of 99.7% or higher), so all the difficulty is in the clothes-changing setting.'),
  B('Each number comes from a single run. The differences between models are of a similar size to the changes we saw between evaluations of the same run, so the planned extra runs are needed before drawing firm conclusions.'),
  H2('1.5 Structure of the report'),
  P('Section 2 explains the metrics and the building blocks shared by the models. Section 3 describes PRCC. Section 4 describes the three models. Section 5 gives the experimental setup and the common protocol. Section 6 presents the results, Section 7 discusses them, and Section 8 concludes. The appendices contain every evaluation from every run, the original reproduction on Market-1501, and the steps needed to repeat the experiments.'),
);

// ---------- 2 background
add(
  H1('2 Background'),
  H2('2.1 Problem formulation'),
  P('A ReID dataset is split into training and test identities that do not overlap, so the model is always tested on people it has never seen. A network f maps each image x to a feature vector f(x). For each query q, all gallery images g are ranked by the cosine distance d(q, g) = 2 - 2 cos(f(q), f(g)). Gallery images of the same person taken by the same camera as the query are ignored, because matching them is trivial.'),
  H2('2.2 Evaluation metrics'),
  B('**Rank-k accuracy.** The percentage of queries for which at least one correct match is among the top k gallery images. We report Rank-1, Rank-5 and Rank-10.'),
  B('**Mean average precision (mAP).** For each query, the average precision over all of its correct matches, averaged over all queries. It rewards finding every correct image, not only the first one.'),
  B('**Mean inverse negative penalty (mINP).** For each query, the number of correct matches divided by the rank of the last correct match, averaged over queries. It shows how far down the list one has to look to find all images of a person.'),
  H2('2.3 Building blocks used by the models'),
  B('**ImageNet pre-training.** All backbones start from weights trained on ImageNet.'),
  B('**Identity loss.** The training identities are treated as classes, and a classifier on top of the feature is trained with cross-entropy, sometimes with label smoothing.'),
  B('**Triplet loss with batch-hard mining.** Each batch holds P identities with K images each. For every image, the loss takes the farthest image of the same person and the closest image of another person in the batch, and pushes the second one further away than the first.'),
  B('**BNNeck.** A batch normalisation layer between the feature and the identity classifier, so that the triplet loss and the identity loss work well together.'),
  B('**Random erasing.** A random rectangle of the training image is blanked out, which imitates occlusion.'),
);

// ---------- 3 dataset
add(
  H1('3 Dataset: PRCC'),
  P('PRCC (Person Re-identification under moderate Clothing Change) contains 33,698 indoor images of 221 people, recorded by three cameras called A, B and C. Cameras A and B see each person in the same outfit. Camera C sees the same person on a different day in different clothes. The dataset also provides contour sketches, which we do not use: all three models work on RGB images only.'),
  P('The training set has 150 people. Each person has two outfits, one seen by cameras A and B and one seen by camera C, which gives 300 outfit classes. The test set has 71 different people. Camera A forms the gallery. Camera B images are the *same-clothes* queries and camera C images are the *clothes-changing* queries. The clothes-changing setting is the task of this report, and the same-clothes setting is reported as a reference. The 5,002 validation images are not used. Table 1 shows the numbers measured by our data loader.'),
  ...table('PRCC statistics (RGB images), measured on the Kaggle copy of the dataset.',
    ['Split', 'Images', 'People', 'Cameras', 'Outfits per person'],
    [['Train', '17,896', '150', '3 (A, B, C)', '2 (300 in total)'],
     ['Validation (not used)', '5,002', 'n/a', '3', 'n/a'],
     ['Gallery (camera A)', '3,384', '71', '1', '1'],
     ['Same-clothes query (camera B)', '3,873', '71', '1', '1'],
     ['Clothes-changing query (camera C)', '3,543', '71', '1', '1']],
    [3000, 1300, 1300, 1500, 1926]),
  P('All results use the full gallery and single-query evaluation without re-ranking, as in the official CAL code. Figure 1 shows one training identity in its two outfits, and Figure 2 shows how the two test settings are built for one test identity.'),
  ...figure('samples_prcc.png', 'One PRCC training identity. The first images show the outfit seen by cameras A and B, the last ones the outfit seen by camera C.', 6.0),
  ...figure('samples_prcc_protocols.png', 'The two PRCC test settings for one test identity: gallery images from camera A, same-clothes queries from camera B and clothes-changing queries from camera C.', 6.0),
);

// ---------- 4 models
add(
  H1('4 The Three Models'),
  H2('4.1 ResNet-50 baseline'),
  P('The baseline is the network used by the CAL paper, trained without any clothes information. It is a ResNet-50 in which the down-sampling of the last stage is removed (last stride 1), so the final feature map keeps more spatial detail. Global max pooling and global average pooling are concatenated into a 4,096-dimensional vector, followed by batch normalisation. A linear identity classifier is trained with label-smoothed cross-entropy. This model is useful because it differs from CAL in one thing only, the clothes losses, so it shows directly what those losses add.'),
  H2('4.2 TransReID'),
  P('*He, S., Luo, H., Wang, P., Wang, F., Li, H. and Jiang, W. TransReID: Transformer-based Object Re-Identification. ICCV 2021.*'),
  P('TransReID uses a vision transformer (ViT-B/16) instead of a convolutional network. The image is cut into 16 x 16 patches, each patch becomes a token, and twelve self-attention layers relate every patch to every other patch. The authors use overlapping patches with a stride of 12, which gives 21 x 11 = 231 patches for a 256 x 128 image. They add two modules designed for ReID:'),
  B('**Side Information Embedding (SIE).** A learned vector for each camera is added to every token, so the network can separate the effect of the camera from the appearance of the person.'),
  B('**Jigsaw Patch Module (JPM).** The last layer is copied into a second branch. In that branch the patch tokens are shifted and shuffled, split into four groups, and each group produces a local feature. Each local feature mixes patches from different parts of the body, which makes the model less sensitive to how well the person is aligned in the box.'),
  P('Each of the five features (one global and four local) has its own BNNeck, identity classifier and triplet loss. At test time the global feature and the four local features are concatenated into a 3,840-dimensional vector. TransReID was designed for the normal same-clothes setting and its paper does not test clothes changes.'),
  ...figure('fig_transreid_architecture.png', 'TransReID as implemented in this study (ViT-B/16, stride 12, SIE and JPM).'),
  H2('4.3 CAL'),
  P('*Gu, X., Chang, H., Ma, B., Bai, S., Shan, S. and Chen, X. Clothes-Changing Person Re-identification with RGB Modality Only. CVPR 2022.*'),
  P('CAL uses the same network as the baseline and adds two parts during training. Each training image gets a clothes label, one class per outfit of each person (300 classes in PRCC).'),
  B('**Clothes classifier.** A cosine classifier learns to predict the outfit from the feature. Its gradient does not reach the backbone, so it only measures how much clothing information the features still contain.'),
  B('**Clothes-based adversarial loss.** The backbone is trained against this classifier. For each image, the loss pulls the feature towards all outfits of the same person at once and away from the outfits of other people. When it succeeds, the classifier can no longer tell the outfits of one person apart, and the network has to identify people by something other than their clothes.'),
  EQ('L_CA = - sum over k in S+ of q(k) log [ exp(s cos(f, w_k)) / ( exp(s cos(f, w_k)) + sum over j in S- of exp(s cos(f, w_j)) ) ]'),
  P('Here S+ are the outfits of the same person, S- are the outfits of other people, w_k is the weight of outfit k, s = 16, and q(k) puts 0.9 on the true outfit and spreads 0.1 over all outfits of that person. For the first 25 epochs only the identity loss is used. From epoch 26 the clothes classifier and the adversarial loss are switched on.'),
  ...figure('fig_cal_architecture.png', 'CAL as implemented in this study. The shaded clothes classifier is trained against the backbone. Without the two clothes losses, this is the baseline of Section 4.1.'),
);

// ---------- 5 setup
add(
  H1('5 Experimental Setup'),
  H2('5.1 Hardware and code'),
  P('All runs used Kaggle notebooks with two NVIDIA Tesla T4 GPUs (16 GB each), PyTorch 2.11 with CUDA 12.8, torchvision and timm. TransReID used both GPUs through data parallelism, and the two ResNet-50 models used one GPU. Mixed precision (FP16) was used for speed, while the exponential terms of the CAL losses were computed in FP32. The three models were implemented in one notebook and checked against the official code of both papers. The pre-trained weights are the ones the papers use: ViT-B/16 trained on ImageNet-21k and fine-tuned on ImageNet-1k for TransReID, and the torchvision ImageNet ResNet-50 for the baseline and CAL. All three fair runs were done in one Kaggle job, one after the other, with random seed 0.'),
  H2('5.2 The common protocol'),
  P('A fair comparison has to decide what is kept equal and what is allowed to differ. We kept equal everything that describes the *experiment*, and let each model keep what describes the *method*. Changing a method\'s own loss or optimiser would mean testing a different method. Table 2 lists the split.'),
  ...table('What the common protocol fixes and what each model keeps.',
    ['Kept the same for all three models', 'Kept from each model\'s own paper'],
    [['Training data: the 17,896 PRCC training images', 'Backbone: ResNet-50 or ViT-B/16 and its pre-trained weights'],
     ['Input size 256 x 128, bilinear resizing', 'Losses: cross-entropy, triplet, JPM and the CAL losses'],
     ['Batch of 64 images: 16 people x 4 images each', 'Optimiser and learning-rate schedule'],
     ['Augmentation: horizontal flip, 10-pixel padding with random crop, random erasing (p = 0.5)', 'Input normalisation that matches the pre-trained weights'],
     ['60 training epochs', 'For CAL: adversarial loss switched on at epoch 26'],
     ['Test: features of the image and of its mirror image added together, cosine distance, full gallery', ''],
     ['Evaluation every 10 epochs, and the result of the **last** epoch is reported', '']],
    [4800, 4226]),
  P('Reporting the last epoch is an important part of the protocol. PRCC has no separate validation identities, and both papers report the best evaluation on the test set. Picking the best epoch on the test set favours models whose score jumps around, because the highest jump is kept. Using the last epoch avoids that, and we still show the whole training curve so that the behaviour during training is visible.'),
  H2('5.3 Settings of each model'),
  ...table('Per-model settings under the common protocol.',
    ['Setting', 'Baseline', 'TransReID', 'CAL'],
    [['Backbone', 'ResNet-50, last stride 1', 'ViT-B/16, stride 12, SIE, JPM', 'ResNet-50, last stride 1'],
     ['Feature size at test', '4,096', '3,840', '4,096'],
     ['Losses', 'cross-entropy with label smoothing 0.1', 'cross-entropy + soft-margin triplet on 5 features', 'baseline loss + clothes classifier + adversarial loss from epoch 26'],
     ['Optimiser', 'Adam, lr 3.5e-4, weight decay 5e-4', 'SGD, momentum 0.9, lr 0.008, weight decay 1e-4', 'Adam, lr 3.5e-4, weight decay 5e-4'],
     ['Learning-rate schedule', 'x 0.1 at epochs 20 and 40', '5 warm-up epochs, then cosine decay', 'x 0.1 at epochs 20 and 40'],
     ['Input normalisation', 'ImageNet mean and std', 'mean and std 0.5', 'ImageNet mean and std'],
     ['GPUs', '1', '2', '1']],
    [2000, 2250, 2526, 2250]),
  H2('5.4 What changed compared with the original settings'),
  P('The earlier reproduction used each paper\'s own settings. Table 4 shows what the common protocol changed for each model. The baseline and CAL changed the most: smaller images, a larger batch with more people and fewer images per person, and a different crop. TransReID kept almost all of its settings but was trained for half as many epochs.'),
  ...table('Original settings against the common protocol.',
    ['Setting', 'Baseline and CAL, original', 'TransReID, original', 'Common protocol'],
    [['Input size', '384 x 192', '256 x 128', '256 x 128'],
     ['Batch', '32 = 4 people x 8 images', '64 = 16 people x 4 images', '64 = 16 people x 4 images'],
     ['Crop augmentation', 'enlarge by 1/8, random crop (p = 0.5)', 'pad 10, random crop', 'pad 10, random crop'],
     ['Epochs', '60', '120', '60'],
     ['Flip at test', 'yes', 'no', 'yes'],
     ['Reported epoch', 'best on test set', 'best on test set', 'last']],
    [1900, 2550, 2200, 2376]),
);

// ---------- 6 results
add(
  H1('6 Results'),
  H2('6.1 Check of the earlier reproduction'),
  P('Before comparing the models, it is worth confirming that the implementation is correct. Table 5 summarises the earlier runs with each paper\'s own settings. TransReID on Market-1501 and CAL on PRCC are both within two points of the published numbers, and our baseline is very close to the baseline reported in the CAL paper. This shows that the data loading, the training code and the evaluation are working as intended. Full details are in Appendix B.'),
  ...table('Earlier reproduction with each paper\'s own settings (%), best evaluation as in the papers.',
    ['Model and dataset', 'Setting', 'Ours R1', 'Ours mAP', 'Paper R1', 'Paper mAP'],
    [['TransReID, Market-1501', 'standard', '95.07', '88.89', '95.2', '88.9'],
     ['CAL, PRCC', 'clothes-changing', '53.26', '54.93', '55.2', '55.8'],
     ['Baseline, PRCC', 'clothes-changing', '45.70', '45.75', '45.6', '43.3'],
     ['TransReID, PRCC', 'clothes-changing', '48.60', '50.41', 'not reported', 'not reported']],
    [2426, 1700, 1200, 1200, 1250, 1250]),
  H2('6.2 Main result: the three models under the common protocol'),
  P('Table 6 gives the final result of each model after 60 epochs under the common protocol. Figure 5 shows the clothes-changing numbers as a chart.'),
  ...table('PRCC results under the common protocol, epoch 60, seed 0 (%). SC = same clothes, CC = clothes-changing.',
    ['Model', 'SC R1', 'SC mAP', 'CC R1', 'CC R5', 'CC R10', 'CC mAP', 'CC mINP'],
    [['**Baseline (ResNet-50)**', '99.90', '**99.68**', '**51.03**', '**54.33**', '**55.69**', '**51.67**', '**36.74**'],
     ['TransReID (ViT-B/16)', '**99.92**', '99.08', '45.67', '51.23', '53.77', '48.28', '34.79'],
     ['CAL (ResNet-50)', '99.74', '98.41', '46.32', '53.09', '55.21', '48.04', '28.82']],
    [2426, 940, 940, 940, 940, 940, 940, 960]),
  ...figure('fig_fair_comparison.png', 'Clothes-changing Rank-1, mAP and mINP on PRCC under the common protocol at epoch 60.', 5.8),
  P('When clothes stay the same, the three models are practically equal: all of them find the right person first in more than 99.7% of cases. When clothes change, accuracy drops by about half for every model. The baseline is the best of the three on every clothes-changing metric. It is 5.36 points of Rank-1 and 3.39 points of mAP ahead of TransReID, and 4.71 points of Rank-1 and 3.63 points of mAP ahead of CAL. TransReID and CAL end very close to each other (48.28% against 48.04% mAP). TransReID has a clearly higher mINP than CAL (34.79% against 28.82%), which means it ranks the hardest correct images of a person higher.'),
  H2('6.3 Behaviour during training'),
  P('The final numbers hide an interesting pattern. Figure 6 shows the clothes-changing mAP and Rank-1 every 10 epochs.'),
  ...figure('fig_fair_curves.png', 'Clothes-changing mAP (left) and Rank-1 (right) on PRCC during training under the common protocol. The vertical line marks where the CAL losses are switched on.', 6.4),
  B('**CAL.** For the first 25 epochs CAL is the same network as the baseline, but it is slightly behind it (39.16% against 42.89% mAP at epoch 10). This is just a difference between two random training runs, since nothing about CAL is active yet. When the clothes losses are switched on, the training loss jumps from 0.85 to 3.40, and within five epochs the clothes-changing mAP rises from 42.65% to 51.45%. At epoch 30 CAL is the best model, 1.50 points of mAP and 2.34 points of Rank-1 ahead of the baseline. After that it slowly declines, and it ends at 48.04%.'),
  B('**Baseline.** The baseline improves steadily, with its biggest step after the learning-rate drop at epoch 20. It reaches its highest point at epoch 50 (51.88% mAP) and stays there.'),
  B('**TransReID.** TransReID learns the training set very quickly: its loss is below 0.05 after 25 epochs and 0.018 at the end. Its clothes-changing mAP is already 47.33% at epoch 10 and then stays between 48.1% and 48.6%. More training makes it fit the training identities better but does not help with new people in new clothes.'),
  H2('6.4 Effect of the protocol on each model'),
  P('Table 7 compares each model under its own settings and under the common protocol. To make the comparison like for like, the original runs are also read at epoch 60 instead of at their best epoch.'),
  ...table('Clothes-changing results at epoch 60: each paper\'s own settings against the common protocol (%).',
    ['Model', 'Own settings R1', 'Own settings mAP', 'Common protocol R1', 'Common protocol mAP', 'Change in mAP'],
    [['Baseline (ResNet-50)', '46.57', '45.56', '51.03', '51.67', '+6.11'],
     ['TransReID (ViT-B/16)', '49.03', '50.11', '45.67', '48.28', '-1.83'],
     ['CAL (ResNet-50)', '52.81', '54.23', '46.32', '48.04', '-6.19'],
     ['CAL minus baseline', '+6.24', '+8.67', '-4.71', '-3.63', '']],
    [2326, 1340, 1340, 1340, 1340, 1340]),
  ...figure('fig_protocol_effect.png', 'Clothes-changing mAP at epoch 60 for each model, trained with its own paper\'s settings and with the common protocol.', 5.8),
  P('The protocol moves the baseline and CAL in opposite directions by about six points each. The baseline gains from the common protocol, and CAL loses. Because these two models share the same network, optimiser and schedule, the result means that the benefit of the CAL loss depends on the training setup it is used with. TransReID changes little. For the original TransReID run, epoch 60 is in the middle of its 120-epoch schedule, so this comparison is only approximate for that model, but its final result after 120 epochs (50.41% mAP) is also close to the 48.28% obtained here.'),
  H2('6.5 Training cost'),
  ...table('Training cost under the common protocol (2 x Tesla T4).',
    ['Model', 'GPUs', 'Time per epoch', 'Training time (60 epochs)', 'Time per test', 'Total job time'],
    [['Baseline (ResNet-50)', '1', '56 s', '1.03 h', '29 s', '68 min'],
     ['TransReID (ViT-B/16)', '2', '180 s', '2.99 h', '75 s', 'about 3.1 h'],
     ['CAL (ResNet-50)', '1', '58 s', '0.97 h', '30 s', '62 min']],
    [2426, 900, 1300, 1700, 1300, 1400]),
  P('TransReID is the most expensive model. It needs about three times the training time of the ResNet-50 models while using twice as many GPUs. CAL costs almost nothing extra over the baseline, because the clothes classifier is a single small matrix.'),
  H2('6.6 Retrieval examples'),
  P('Figures 8 to 10 show five clothes-changing queries and the top ten gallery images returned by each model. The same five queries are used for all three models. A thick solid frame and a tick mark a correct match, and a thin dashed frame and a cross mark a wrong one.'),
  ...figure('retrieval_fair_baseline_s0.png', 'Baseline: top ten matches for five clothes-changing queries.', 6.3),
  ...figure('retrieval_fair_transreid_s0.png', 'TransReID: top ten matches for the same five queries.', 6.3),
  ...figure('retrieval_fair_cal_s0.png', 'CAL: top ten matches for the same five queries.', 6.3),
  P('The examples show the typical behaviour in this setting. For some queries all ten matches are correct even though the outfit is different, so the models do use cues other than clothes. For other queries the top results are all wrong and show a different person in clothes of a similar colour or pattern, which is the clothing shortcut the task is designed to test.'),
);

// ---------- 7 discussion
add(
  H1('7 Discussion'),
  H2('7.1 Why did the baseline end on top?'),
  P('The result is surprising, because the CAL paper and our own earlier run both show CAL well ahead of the baseline. We did not run extra experiments to isolate the cause, so the points below are possible explanations, not proven ones.'),
  B('**Image size.** The common protocol uses 256 x 128 images instead of the 384 x 192 used by CAL. Once clothing information is pushed out of the features, CAL has to rely on smaller details such as hair, face and shoes. These details are harder to see in smaller images, so CAL may have less to work with than in its original setting.'),
  B('**Batch structure.** The original CAL batch has 4 people with 8 images each, while the common protocol uses 16 people with 4 images each. With 8 images per person, a batch more often contains both outfits of the same person. This may matter for a loss whose whole purpose is to make the outfits of one person look alike.'),
  B('**Instability after the switch.** CAL is clearly better than the baseline five epochs after its losses start, and then loses ground. Adversarial training is a contest between two parts of the model, and it can drift over time. With the original settings CAL stayed stable, but with the new ones it did not. Without model selection on the test set, this drift shows up in the final number.'),
  B('**The baseline gained from the new settings.** The baseline improved by six points of mAP. More people per batch and the padding crop may simply suit a model trained with the identity loss alone. Part of the gap between CAL and the baseline is therefore a better baseline, not only a worse CAL.'),
  H2('7.2 TransReID on clothes-changing data'),
  P('TransReID is almost perfect when clothes stay the same, but it loses about half of its accuracy when they change, in both the earlier run and this one. Its training loss goes almost to zero, so it separates the 150 training people very well. In PRCC, two thirds of each person\'s training images show the same outfit, and nothing in its training objective discourages it from using clothing to tell people apart. The stronger backbone with global attention does not by itself change what the model learns to look at. Its clothes-changing score also stopped improving after about 10 to 20 epochs, which suggests that longer training would not help.'),
  H2('7.3 What "fair" means here'),
  P('No single protocol is fair to every method. The common protocol is fair in the sense that every model gets the same data, the same images, the same budget and the same test. However, each method was originally tuned for its own settings, and the common settings are close to TransReID\'s original ones and further from CAL\'s. A method that loses accuracy under a new protocol is not necessarily a weak method, but it is less robust to the training setup. For a practical system this is useful information, because the setup in practice is rarely the one used in a paper. We therefore think both views are worth reporting: the published-settings results show what each method can reach, and the common protocol shows how they compare on equal terms.'),
  H2('7.4 Limitations'),
  B('**One run per model.** Each number comes from a single training run with seed 0. Within one run, the clothes-changing mAP of CAL moved by up to 3.4 points between the last four evaluations, which is about the size of the gaps between the models. Two more runs per model (seeds 1 and 2) are planned, so that each result can be given as a mean and a standard deviation.'),
  B('**One protocol.** We tested one shared protocol. Repeating the comparison with CAL\'s original settings (384 x 192, 4 people x 8 images) for all three models would show whether the ranking depends on which method the protocol is closer to.'),
  B('**One dataset.** LTCC, the second clothes-changing dataset in the CAL paper, is only available after signing a data agreement and was not available in time.'),
  B('**No validation set.** PRCC has no separate validation identities. We avoided choosing the best epoch on the test set, but the test set was still used to draw the training curves.'),
);

// ---------- 8 conclusion
add(
  H1('8 Conclusion'),
  P('We compared a ResNet-50 baseline, TransReID and CAL on one task, clothes-changing person re-identification on PRCC, and retrained all three under one training protocol. Under this protocol the plain baseline gave the best final result (51.03% Rank-1, 51.67% mAP), while TransReID (45.67%, 48.28%) and CAL (46.32%, 48.04%) ended about 3.5 points of mAP lower. CAL improved quickly as soon as its adversarial loss was switched on and was the best model at epoch 30, but the gain did not last until the end of training.'),
  P('With each paper\'s own settings, CAL was the best model by a wide margin, and our earlier reproduction matched the published numbers. Taken together, the two sets of results show that the benefit of the CAL loss is real but depends on the training setup, while a larger transformer backbone on its own does not solve the clothes-changing problem. The next steps are to repeat each run with two more seeds, to repeat the comparison under CAL\'s original settings, and to test whether the CAL loss works better on top of the TransReID backbone.'),
);

// ---------- references
add(
  H1('References'),
  ...[
    'He, S., Luo, H., Wang, P., Wang, F., Li, H. and Jiang, W. TransReID: Transformer-based Object Re-Identification. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021. arXiv:2102.04378.',
    'Gu, X., Chang, H., Ma, B., Bai, S., Shan, S. and Chen, X. Clothes-Changing Person Re-identification with RGB Modality Only. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. arXiv:2204.06890.',
    'Yang, Q., Wu, A. and Zheng, W.-S. Person Re-identification by Contour Sketch under Moderate Clothing Change. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2019.',
    'Zheng, L., Shen, L., Tian, L., Wang, S., Wang, J. and Tian, Q. Scalable Person Re-identification: A Benchmark. In ICCV, 2015.',
    'Qian, X., Wang, W., Zhang, L., Zhu, F., Fu, Y., Xiang, T., Jiang, Y.-G. and Xue, X. Long-Term Cloth-Changing Person Re-identification. In ACCV, 2020.',
    'Dosovitskiy, A. et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In ICLR, 2021.',
    'He, K., Zhang, X., Ren, S. and Sun, J. Deep Residual Learning for Image Recognition. In CVPR, 2016.',
    'Luo, H., Gu, Y., Liao, X., Lai, S. and Jiang, W. Bag of Tricks and a Strong Baseline for Deep Person Re-identification. In CVPR Workshops, 2019.',
    'Hermans, A., Beyer, L. and Leibe, B. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737, 2017.',
    'Ye, M., Shen, J., Lin, G., Xiang, T., Shao, L. and Hoi, S. C. H. Deep Learning for Person Re-identification: A Survey and Outlook. IEEE TPAMI, 2021.',
    'Zhong, Z., Zheng, L., Kang, G., Li, S. and Yang, Y. Random Erasing Data Augmentation. In AAAI, 2020.',
    'Official code: github.com/damo-cv/TransReID and github.com/guxinqian/Simple-CCReID.',
  ].map((r, i) => P(`[${i + 1}] ${r}`, { alignment: AlignmentType.LEFT })),
);

// ---------- appendix A: logs of the fair runs
const evalRows = (rows) => rows.map(r => r.map((v, i) => i === 0 ? String(v) : v.toFixed(2)));
const H8 = ['Epoch', 'SC R1', 'SC mAP', 'CC R1', 'CC R5', 'CC R10', 'CC mAP', 'CC mINP'];
add(
  H1('Appendix A: Evaluation Logs of the Common-Protocol Runs'),
  P('Every test evaluation recorded during the three runs of Section 6 (%). SC = same clothes, CC = clothes-changing. The last row of each table is the reported result.'),
  ...table('Baseline (ResNet-50), common protocol, seed 0.', H8,
    evalRows([[10, 99.12, 96.31, 43.04, 49.25, 51.45, 42.89, 22.07], [20, 98.42, 96.80, 45.84, 49.87, 52.33, 44.16, 22.41],
              [30, 99.97, 99.47, 49.11, 52.41, 54.28, 49.95, 33.53], [40, 99.82, 99.47, 49.20, 53.66, 55.72, 50.61, 35.29],
              [50, 99.97, 99.67, 51.11, 54.84, 57.18, 51.88, 37.15], [60, 99.90, 99.68, 51.03, 54.33, 55.69, 51.67, 36.74]]),
    even(8), { centerFirst: true }),
  ...table('TransReID (ViT-B/16), common protocol, seed 0.', H8,
    evalRows([[10, 99.56, 98.75, 42.68, 49.25, 52.47, 47.33, 33.50], [20, 100.00, 98.97, 46.46, 53.46, 55.91, 48.62, 33.81],
              [30, 99.77, 98.99, 44.71, 50.24, 53.26, 48.13, 33.76], [40, 99.87, 99.06, 46.46, 51.45, 53.94, 48.62, 35.44],
              [50, 99.92, 99.13, 45.81, 51.14, 53.97, 48.29, 35.03], [60, 99.92, 99.08, 45.67, 51.23, 53.77, 48.28, 34.79]]),
    even(8), { centerFirst: true }),
  ...table('CAL (ResNet-50), common protocol, seed 0.', H8,
    evalRows([[10, 97.83, 95.35, 40.45, 45.75, 48.29, 39.16, 19.38], [20, 97.91, 95.58, 41.66, 47.39, 50.80, 42.65, 22.44],
              [30, 100.00, 98.93, 51.45, 55.72, 58.06, 51.45, 32.36], [40, 99.97, 98.85, 49.51, 53.91, 55.52, 49.75, 31.54],
              [50, 99.85, 98.52, 48.12, 54.50, 56.14, 49.64, 31.11], [60, 99.74, 98.41, 46.32, 53.09, 55.21, 48.04, 28.82]]),
    even(8), { centerFirst: true }),
);

// ---------- appendix B: earlier reproduction with original settings
add(
  H1('Appendix B: Earlier Reproduction With Each Paper\'s Own Settings'),
  P('This appendix records the earlier stage of the project, in which each method was trained with the settings of its own paper. These runs are the source of Table 5 and of the "own settings" columns of Table 7.'),
  H2('B.1 TransReID on Market-1501'),
  P('Market-1501 has 1,501 people recorded by six cameras outside a supermarket, with 12,936 training images of 751 people, 3,368 query images and 15,913 gallery images of 750 other people. Everyone wears the same clothes in every camera. TransReID was trained for 120 epochs with its official settings and reached 95.07% Rank-1 and 88.89% mAP, against 95.2% and 88.9% in the paper.'),
  ...figure('fig_transreid_market_curve.png', 'TransReID on Market-1501 during training. Dotted lines are the values reported in the paper.', 5.6),
  ...table('TransReID on Market-1501, own settings (%).', ['Epoch', 'Rank-1', 'Rank-5', 'Rank-10', 'mAP', 'mINP'],
    evalRows([[20, 93.38, 97.57, 98.55, 84.77, 60.75], [40, 94.18, 97.95, 98.81, 87.25, 65.52], [60, 94.27, 98.04, 98.99, 87.51, 66.04],
              [80, 94.86, 98.22, 98.93, 88.56, 67.79], [100, 94.89, 98.25, 98.90, 88.84, 68.54], [120, 95.07, 98.19, 98.96, 88.89, 68.58]]),
    even(6), { centerFirst: true }),
  H2('B.2 Baseline, CAL and TransReID on PRCC'),
  P('The baseline and CAL were trained for 60 epochs with the official CAL settings (384 x 192 images, 4 people x 8 images per batch). TransReID was trained for 120 epochs with its Market-1501 settings. Figure 12 shows the clothes-changing mAP during these runs.'),
  ...figure('fig_prcc_cc_curves.png', 'Clothes-changing mAP on PRCC with each paper\'s own settings.', 5.6),
  ...table('Baseline on PRCC, own settings (%).', H8,
    evalRows([[5, 96.15, 86.65, 38.92, 46.43, 49.51, 32.25, 10.23], [10, 98.06, 94.03, 39.60, 43.66, 45.81, 36.55, 15.13],
              [15, 98.58, 94.33, 34.04, 38.27, 40.84, 33.15, 13.02], [20, 98.37, 94.37, 35.85, 43.16, 46.18, 33.30, 9.16],
              [25, 99.74, 96.77, 42.79, 46.99, 49.28, 40.26, 13.86], [30, 99.64, 96.98, 44.96, 50.30, 52.81, 42.50, 16.56],
              [35, 99.69, 97.78, 44.20, 48.94, 52.07, 43.44, 17.57], [40, 99.51, 97.57, 43.66, 48.86, 52.02, 43.26, 18.84],
              [45, 99.77, 97.97, 45.70, 50.64, 54.05, 45.75, 22.26], [50, 99.82, 98.01, 46.54, 51.03, 53.66, 45.43, 20.80],
              [55, 99.69, 97.93, 46.88, 51.51, 54.11, 45.68, 22.29], [60, 99.74, 97.97, 46.57, 51.65, 54.22, 45.56, 19.60]]),
    even(8), { centerFirst: true }),
  ...table('CAL on PRCC, own settings (%).', H8,
    evalRows([[5, 94.89, 86.33, 34.10, 40.76, 44.23, 29.06, 8.53], [10, 96.54, 89.52, 36.55, 42.93, 46.32, 33.59, 13.31],
              [15, 99.33, 95.09, 45.39, 50.83, 53.46, 41.85, 14.05], [20, 97.19, 92.79, 45.55, 51.37, 55.04, 41.50, 13.07],
              [25, 99.35, 96.47, 45.98, 51.03, 53.60, 42.81, 13.25], [30, 100.00, 99.43, 50.55, 55.77, 57.95, 51.47, 34.32],
              [35, 100.00, 99.48, 53.32, 58.54, 61.11, 54.36, 36.30], [40, 100.00, 99.63, 53.82, 57.32, 58.62, 54.00, 36.84],
              [45, 100.00, 99.66, 52.72, 56.51, 58.23, 54.19, 36.74], [50, 100.00, 99.66, 52.19, 57.86, 60.32, 53.57, 35.93],
              [55, 100.00, 99.73, 53.26, 57.66, 59.75, 54.93, 38.18], [60, 100.00, 99.75, 52.81, 56.59, 58.85, 54.23, 36.98]]),
    even(8), { centerFirst: true }),
  ...table('TransReID on PRCC, own settings (%).', H8,
    evalRows([[20, 100.00, 99.10, 48.66, 54.39, 56.62, 49.84, 32.73], [40, 99.97, 99.29, 46.63, 53.82, 56.48, 48.67, 33.71],
              [60, 99.95, 99.12, 49.03, 54.78, 58.26, 50.11, 33.36], [80, 99.95, 99.19, 48.91, 54.76, 57.92, 50.12, 34.20],
              [100, 99.97, 99.11, 48.74, 54.25, 56.70, 50.24, 33.79], [120, 99.97, 99.12, 48.60, 54.61, 57.24, 50.41, 34.04]]),
    even(8), { centerFirst: true }),
);

// ---------- appendix C: how to repeat
add(
  H1('Appendix C: Repeating the Experiments'),
  P('All code is in one Kaggle notebook with nine cells (file reid_kaggle_notebook.py, also available as separate cell files). The datasets are attached as Kaggle inputs: PRCC (lyqassf/prccdata1) and Market-1501 (iiierie/market-1501-v15-09-15).', { alignment: AlignmentType.LEFT }),
  N('Create a Kaggle notebook with GPU T4 x2 and internet switched on, and attach the datasets.', 'numbers2'),
  N('Paste the nine cells. Cell 1 must start with the line %%writefile /kaggle/working/reid_common.py.', 'numbers2'),
  N('In Cell 2 set JOB = \'fair_all\', QUICK = False and SEED = 0. The common protocol is defined in the FAIR settings in the same cell.', 'numbers2'),
  N('Start the run with Save Version, then Save and Run All. The three models are trained one after the other in about 5.5 hours.', 'numbers2'),
  N('Cell 7 writes the results table (fair_comparison_prcc.csv) and the charts, Cell 8 writes the retrieval examples, and Cell 9 packs all result files into one zip file.', 'numbers2'),
  N('For more seeds, attach the previous version as an input and run again with SEED = 1 and SEED = 2. Cell 7 then reports the mean and standard deviation over all finished seeds.', 'numbers2'),
);

// ------------------------------------------------------------------ document
const doc = new Document({
  creator: 'ReID study', title: 'Clothes-Changing Person Re-Identification on PRCC', description: 'Comparison report',
  styles: {
    default: { document: { run: { font: FONT, size: 24, color: BLACK } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 32, bold: true, color: BLACK, font: FONT }, paragraph: { spacing: { before: 240, after: 200 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 26, bold: true, color: BLACK, font: FONT }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1, keepNext: true } },
      { id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 24, bold: true, color: BLACK, font: FONT }, paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: 'bullets', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: 'numbers2', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: 'numbers', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 20, color: BLACK })] })] }) },
    children: body,
  }],
});

Packer.toBuffer(doc).then(buf => { fs.writeFileSync(OUT, buf); console.log('written', OUT, `${figNo} figures, ${tabNo} tables`); });
