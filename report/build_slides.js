// Builds the presentation for the PRCC fair-comparison report. Palette: oxblood, mustard, olive (fabric colours).
const fs = require('fs');
const path = require('path');
const pptxgen = require('pptxgenjs');
const { applyTheme } = require(path.join(process.env.PPTX_SKILL_DIR, 'scripts', 'apply_theme.js'));

const FIG = path.join(__dirname, 'figures');
const OUT = path.join(__dirname, process.env.DECK_NAME || 'ReID_Presentation_Fair_Comparison_PRCC.pptx');

const THEME = {
  name: 'Fabric ReID', headFontFace: 'Cambria', bodyFontFace: 'Calibri',
  colors: { dk1: '1F1416', lt1: 'FFFFFF', dk2: '2A1619', lt2: 'F4F1F0', accent1: '8C2131', accent2: 'D49A1F',
            accent3: '5E6B24', accent4: 'B8576A', accent5: 'E8C36A', accent6: '6E6264', hlink: '8C2131', folHlink: '5E6B24' },
};
const HEX = { oxblood: '8C2131', mustard: 'D49A1F', olive: '5E6B24', ink: '2A1619', grid: 'E3DEDC', line: 'D8CFCD' };
const MODEL = { baseline: HEX.olive, transreid: HEX.mustard, cal: HEX.oxblood };

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';                 // 13.33 x 7.5 in
pres.title = 'Clothes-Changing Person Re-Identification on PRCC';
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const W = 13.33, M = 0.6;

// ------------------------------------------------------------------ layouts
pres.defineSlideMaster({
  title: 'DARK', background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: M, y: 2.1, w: W - 2 * M, h: 1.6, fontSize: 40, bold: true,
      color: C.background1, valign: 'bottom', fontFace: THEME.headFontFace }, text: '' } },
    { placeholder: { options: { name: 'body', type: 'body', x: M, y: 3.9, w: W - 2 * M, h: 2.2, fontSize: 18,
      color: C.background2, valign: 'top' }, text: '' } },
  ],
});
pres.defineSlideMaster({
  title: 'LIGHT', background: { color: C.background1 },
  slideNumber: { x: W - 1.0, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.accent6, align: 'right' },
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: M, y: 0.35, w: W - 2 * M, h: 0.9, fontSize: 32, bold: true,
      color: C.accent1, valign: 'middle', fontFace: THEME.headFontFace }, text: '' } },
    { text: { text: 'Clothes-changing ReID on PRCC', options: { x: M, y: 7.0, w: 6, h: 0.3, fontSize: 10, color: C.accent6, margin: 0 } } },
  ],
});

// ------------------------------------------------------------------ helpers
const png = (f) => { const b = fs.readFileSync(path.join(FIG, f)); return { path: path.join(FIG, f), w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; };
function fitImage(slide, file, x, y, maxW, maxH, name) {
  const im = png(file); const r = Math.min(maxW / im.w, maxH / im.h);
  const w = im.w * r, h = im.h * r;
  slide.addImage({ path: im.path, x: x + (maxW - w) / 2, y: y + (maxH - h) / 2, w, h, objectName: name, altText: name });
}
const text = (slide, t, o) => slide.addText(t, { isTextBox: true, color: C.text1, fontSize: 16, valign: 'top', ...o });
const card = (slide, x, y, w, h, name) => slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: name });
const bullets = (items, o = {}) => items.map((t, i) => ({ text: t, options: { bullet: { indent: 18 }, breakLine: i < items.length - 1, paraSpaceAfter: 8, ...o } }));
function circleNum(slide, n, x, y, d = 0.55, fill = C.accent1) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill }, objectName: `badge ${n}` });
  text(slide, String(n), { x, y, w: d, h: d, fontSize: 18, bold: true, color: C.background1, align: 'center', valign: 'middle', margin: 0 });
}
const chartText = { catAxisLabelFontFace: '+mn-lt', valAxisLabelFontFace: '+mn-lt', dataLabelFontFace: '+mn-lt', legendFontFace: '+mn-lt',
  titleFontFace: '+mn-lt', catAxisLabelColor: '2A1619', valAxisLabelColor: '6E6264', dataLabelColor: '2A1619', legendColor: '2A1619',
  catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, dataLabelFontSize: 11, legendFontSize: 12 };
const light = (sec) => pres.addSlide({ masterName: 'LIGHT', sectionTitle: sec });

// ================================================================== 1 title
pres.addSection({ title: 'Introduction' });
let s = pres.addSlide({ masterName: 'DARK', sectionTitle: 'Introduction' });
s.addText('Clothes-Changing Person Re-Identification on PRCC', { placeholder: 'title' });
s.addText([
  { text: 'A comparison of a ResNet-50 baseline, TransReID and CAL under one training protocol', options: { breakLine: true } },
  { text: ' ', options: { breakLine: true, fontSize: 10 } },
  { text: '[Your name]   |   November 2026', options: { fontSize: 16, color: C.accent2 } },
], { placeholder: 'body' });
s.addNotes('Today I compare three person re-identification models on the same task: finding a person in another camera when they wear different clothes.');

// ================================================================== 2 the task
s = light('Introduction');
s.addText('The task: find the same person in new clothes', { placeholder: 'title' });
text(s, bullets([
  'Re-identification ranks gallery images so the same person comes first',
  'Most models learn to match people by their clothes',
  'In PRCC, camera C shows each person on another day in different clothes',
  'The model has to use body shape, hair, face and posture instead',
]), { x: M, y: 1.5, w: 4.6, h: 4.6, fontSize: 18 });
fitImage(s, 'samples_prcc.png', 5.4, 1.45, 7.33, 4.9, 'PRCC sample identity');
text(s, 'One training identity in its two outfits (cameras A and B against camera C)', { x: 5.4, y: 5.35, w: 7.33, h: 0.4, fontSize: 11, color: C.accent6, align: 'center' });

// ================================================================== 3 why
s = light('Introduction');
s.addText('Why the earlier numbers could not be compared', { placeholder: 'title' });
text(s, 'In the first stage each paper was reproduced with its own settings. They matched the papers, but they were trained differently:', { x: M, y: 1.4, w: W - 2 * M, h: 0.7, fontSize: 17 });
const hdr = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.accent1 } } });
s.addTable([
  [hdr('Setting'), hdr('CAL and baseline (own)'), hdr('TransReID (own)'), hdr('Common protocol')],
  ['Input size', '384 x 192', '256 x 128', '256 x 128'],
  ['Batch', '4 people x 8 images', '16 people x 4 images', '16 people x 4 images'],
  ['Epochs', '60', '120', '60'],
  ['Flip at test', 'yes', 'no', 'yes'],
  ['Reported epoch', 'best on test set', 'best on test set', 'last'],
], { x: M, y: 2.25, w: W - 2 * M, colW: [2.6, 3.4, 3.2, 2.93], fontSize: 15, color: C.text1, border: { type: 'solid', pt: 0.75, color: 'D8CFCD' },
     rowH: 0.52, valign: 'middle', margin: [0.04, 0.12, 0.04, 0.12], objectName: 'settings table' });
card(s, M, 5.6, W - 2 * M, 0.95, 'goal card');
text(s, [{ text: 'Goal: ', options: { bold: true } }, { text: 'one task, one dataset, one protocol, and all three models trained again from scratch.' }],
  { x: M + 0.25, y: 5.6, w: W - 2 * M - 0.5, h: 0.95, fontSize: 18, valign: 'middle' });
s.addNotes('Any difference in the old numbers could come from the settings instead of the methods. So the supervisor asked for one fixed task and retraining.');

// ================================================================== 4 models
pres.addSection({ title: 'Method' });
s = light('Method');
s.addText('Three models, one task', { placeholder: 'title' });
const models = [
  ['Baseline', 'ResNet-50', ['Same network as CAL', 'Identity loss only', 'Shows what the CAL loss adds']],
  ['TransReID', 'ViT-B/16, ICCV 2021', ['Transformer over image patches', 'Camera embedding (SIE)', 'Jigsaw patch branch (JPM)']],
  ['CAL', 'ResNet-50, CVPR 2022', ['Clothes classifier per outfit', 'Adversarial loss hides outfits', 'Switched on at epoch 26']],
];
const modelFill = [C.accent3, C.accent2, C.accent1];
models.forEach(([name, sub, pts], i) => {
  const x = M + i * 4.1, w = 3.8;
  card(s, x, 1.5, w, 4.9, `${name} card`);
  circleNum(s, i + 1, x + 0.3, 1.8, 0.55, modelFill[i]);
  text(s, name, { x: x + 1.0, y: 1.75, w: w - 1.2, h: 0.5, fontSize: 24, bold: true, fontFace: THEME.headFontFace, margin: 0, valign: 'middle', color: modelFill[i] });
  text(s, sub, { x: x + 1.0, y: 2.25, w: w - 1.2, h: 0.4, fontSize: 14, color: C.accent6, margin: 0 });
  text(s, bullets(pts), { x: x + 0.3, y: 3.0, w: w - 0.5, h: 3.2, fontSize: 16 });
});
s.addNotes('The baseline is the CAL network without the clothes losses, so comparing those two isolates the CAL loss.');

// ================================================================== 5 protocol
s = light('Method');
s.addText('The common protocol', { placeholder: 'title' });
card(s, M, 1.45, 5.9, 5.1, 'same card');
text(s, 'Same for all three', { x: M + 0.3, y: 1.6, w: 5.3, h: 0.5, fontSize: 22, bold: true, fontFace: THEME.headFontFace, margin: 0, color: C.accent1 });
text(s, bullets(['17,896 PRCC training images', '256 x 128 input', 'Batch of 16 people x 4 images', 'Flip, pad and crop, random erasing',
  '60 epochs', 'Flip at test, cosine distance', 'Report the last epoch']), { x: M + 0.3, y: 2.25, w: 5.4, h: 4.2, fontSize: 17 });
s.addShape(pres.shapes.RECTANGLE, { x: 6.85, y: 1.45, w: 5.88, h: 5.1, fill: { color: C.background1 }, line: { color: 'D8CFCD', width: 1.5 }, objectName: 'own card' });
text(s, 'Kept from each paper', { x: 7.15, y: 1.6, w: 5.3, h: 0.5, fontSize: 22, bold: true, fontFace: THEME.headFontFace, margin: 0, color: C.accent1 });
text(s, bullets(['Backbone and pre-trained weights', 'Loss functions', 'Optimiser and learning-rate schedule', 'Input normalisation']),
  { x: 7.15, y: 2.25, w: 5.4, h: 2.4, fontSize: 17 });
text(s, [{ text: 'Why: ', options: { bold: true } }, { text: 'changing a method\'s own loss or optimiser would test a different method.' }],
  { x: 7.15, y: 4.8, w: 5.3, h: 1.4, fontSize: 16, color: C.text2 });
s.addNotes('Reporting the last epoch matters: picking the best epoch on the test set favours models whose score jumps around.');

// ================================================================== 6 reproduction check
pres.addSection({ title: 'Results' });
s = light('Results');
s.addText('First check: the reproduction matches the papers', { placeholder: 'title' });
const stats = [['95.07', 'TransReID Rank-1, Market-1501', 'paper 95.2'], ['54.93', 'CAL mAP, PRCC clothes-changing', 'paper 55.8'],
               ['45.70', 'Baseline Rank-1, PRCC clothes-changing', 'paper 45.6']];
stats.forEach(([v, lab, p], i) => {
  const x = M + i * 4.1;
  card(s, x, 1.6, 3.8, 3.3, `stat ${i + 1}`);
  text(s, v, { x: x + 0.2, y: 1.8, w: 3.4, h: 1.3, fontSize: 60, bold: true, fontFace: THEME.headFontFace, align: 'center', valign: 'middle', margin: 0,
    color: [C.accent2, C.accent1, C.accent3][i] });
  text(s, lab, { x: x + 0.2, y: 3.15, w: 3.4, h: 0.9, fontSize: 16, align: 'center', margin: 0 });
  text(s, p, { x: x + 0.2, y: 4.1, w: 3.4, h: 0.5, fontSize: 14, color: C.accent6, align: 'center', margin: 0 });
});
text(s, 'All three are within about two points of the published numbers, so the data loading, training code and evaluation are correct.',
  { x: M, y: 5.35, w: W - 2 * M, h: 0.9, fontSize: 18 });

// ================================================================== 7 main result
s = light('Results');
s.addText('Main result: the baseline ends on top', { placeholder: 'title' });
s.addChart(pres.charts.BAR, [
  { name: 'Baseline', labels: ['Rank-1', 'mAP', 'mINP'], values: [51.03, 51.67, 36.74] },
  { name: 'TransReID', labels: ['Rank-1', 'mAP', 'mINP'], values: [45.67, 48.28, 34.79] },
  { name: 'CAL', labels: ['Rank-1', 'mAP', 'mINP'], values: [46.32, 48.04, 28.82] },
], { x: M, y: 1.4, w: 7.9, h: 5.3, barDir: 'col', barGrouping: 'clustered', chartColors: [MODEL.baseline, MODEL.transreid, MODEL.cal],
     showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0.0', valAxisMinVal: 0, valAxisMaxVal: 60, valAxisMajorUnit: 10,
     valGridLine: { color: 'E3DEDC', size: 0.5 }, catGridLine: { style: 'none' }, showLegend: true, legendPos: 'b',
     showTitle: true, title: 'PRCC clothes-changing, epoch 60 (%)', titleFontSize: 14, titleColor: '2A1619', ...chartText, objectName: 'main result chart' });
card(s, 8.85, 1.5, 3.88, 5.1, 'takeaways card');
text(s, bullets([
  'Baseline: 51.0 Rank-1, 51.7 mAP',
  'TransReID and CAL end about 3.5 mAP lower and almost tied',
  'Same clothes: all three above 99.7 Rank-1',
]), { x: 9.1, y: 1.75, w: 3.45, h: 4.7, fontSize: 17 });
s.addNotes('When clothes stay the same, all three models are nearly perfect. All the difficulty is in the clothes-changing setting.');

// ================================================================== 8 curves
s = light('Results');
s.addText('CAL wins for a while, then fades', { placeholder: 'title' });
const ep = ['10', '20', '30', '40', '50', '60'];
s.addChart(pres.charts.LINE, [
  { name: 'Baseline', labels: ep, values: [42.89, 44.16, 49.95, 50.61, 51.88, 51.67] },
  { name: 'TransReID', labels: ep, values: [47.33, 48.62, 48.13, 48.62, 48.29, 48.28] },
  { name: 'CAL', labels: ep, values: [39.16, 42.65, 51.45, 49.75, 49.64, 48.04] },
], { x: M, y: 1.4, w: 7.9, h: 5.3, chartColors: [MODEL.baseline, MODEL.transreid, MODEL.cal], lineSize: 3, lineDataSymbolSize: 8,
     lineDataSymbol: 'circle', valAxisMinVal: 36, valAxisMaxVal: 54, valAxisMajorUnit: 2, valGridLine: { color: 'E3DEDC', size: 0.5 },
     catGridLine: { style: 'none' }, showLegend: true, legendPos: 'b', showTitle: true, title: 'Clothes-changing mAP (%) by epoch',
     titleFontSize: 14, titleColor: '2A1619', showCatAxisTitle: true, catAxisTitle: 'epoch', catAxisTitleFontSize: 12, catAxisTitleColor: '595959',
     ...chartText, objectName: 'training curves chart' });
card(s, 8.85, 1.5, 3.88, 5.1, 'curve notes card');
text(s, [
  { text: 'Epoch 26: ', options: { bold: true } }, { text: 'CAL loss switched on', options: { breakLine: true } },
  { text: ' ', options: { breakLine: true, fontSize: 8 } },
  { text: 'Epoch 30: ', options: { bold: true } }, { text: 'CAL is best, 51.45 mAP', options: { breakLine: true } },
  { text: ' ', options: { breakLine: true, fontSize: 8 } },
  { text: 'Epoch 60: ', options: { bold: true } }, { text: 'CAL down to 48.04, baseline still climbing', options: { breakLine: true } },
  { text: ' ', options: { breakLine: true, fontSize: 8 } },
  { text: 'TransReID: ', options: { bold: true } }, { text: 'flat at about 48 from epoch 10' },
], { x: 9.1, y: 1.75, w: 3.45, h: 4.7, fontSize: 17 });

// ================================================================== 9 protocol effect
s = light('Results');
s.addText('The training setup changes the ranking', { placeholder: 'title' });
s.addChart(pres.charts.BAR, [
  { name: "Each paper's own settings", labels: ['Baseline', 'TransReID', 'CAL'], values: [45.56, 50.11, 54.23] },
  { name: 'Common protocol', labels: ['Baseline', 'TransReID', 'CAL'], values: [51.67, 48.28, 48.04] },
], { x: M, y: 1.4, w: 7.9, h: 5.3, barDir: 'col', barGrouping: 'clustered', chartColors: ['B5A9A7', HEX.oxblood], showValue: true,
     dataLabelPosition: 'outEnd', dataLabelFormatCode: '0.0', valAxisMinVal: 0, valAxisMaxVal: 60, valAxisMajorUnit: 10,
     valGridLine: { color: 'E3DEDC', size: 0.5 }, catGridLine: { style: 'none' }, showLegend: true, legendPos: 'b', showTitle: true,
     title: 'Clothes-changing mAP at epoch 60 (%)', titleFontSize: 14, titleColor: '2A1619', ...chartText, objectName: 'protocol effect chart' });
[['+8.67', 'CAL minus baseline, own settings'], ['-3.63', 'CAL minus baseline, common protocol']].forEach(([v, lab], i) => {
  const y = 1.5 + i * 2.6;
  card(s, 8.85, y, 3.88, 2.4, `delta ${i + 1}`);
  text(s, v, { x: 9.0, y: y + 0.2, w: 3.6, h: 1.2, fontSize: 54, bold: true, fontFace: THEME.headFontFace, align: 'center', valign: 'middle', margin: 0,
    color: i === 0 ? C.accent3 : C.accent1 });
  text(s, lab, { x: 9.0, y: y + 1.45, w: 3.6, h: 0.7, fontSize: 15, align: 'center', margin: 0 });
});
s.addNotes('The baseline and CAL share the same network and optimiser, so this shows that the benefit of the CAL loss depends on the training setup.');

// ================================================================== 10 retrieval
s = light('Results');
s.addText('Retrieval examples (CAL)', { placeholder: 'title' });
fitImage(s, 'retrieval_fair_cal_s0.png', M, 1.3, 8.4, 5.55, 'CAL retrieval examples');
card(s, 9.25, 1.5, 3.48, 5.1, 'retrieval notes card');
text(s, bullets([
  'Five clothes-changing queries, top ten matches',
  'Solid frame and tick: correct person',
  'Dashed frame and cross: wrong person',
  'Some queries are found in new clothes, others match a stranger in similar clothes',
]), { x: 9.45, y: 1.75, w: 3.1, h: 4.7, fontSize: 15 });

// ================================================================== 11 why
pres.addSection({ title: 'Discussion' });
s = light('Discussion');
s.addText('Why did the baseline end on top?', { placeholder: 'title' });
text(s, 'Possible explanations, not yet tested one by one:', { x: M, y: 1.3, w: W - 2 * M, h: 0.5, fontSize: 17, color: C.accent6 });
const why = [
  ['Smaller images', 'Without clothes, CAL needs hair, face and shoes, which are harder to see at 256 x 128'],
  ['Batch structure', '4 images per person means a batch less often holds both outfits of one person'],
  ['Unstable after the switch', 'The adversarial contest drifts over time; reporting the last epoch shows the drift'],
  ['A stronger baseline', 'The baseline itself gained 6 mAP from the new settings'],
];
why.forEach(([h, d], i) => {
  const x = M + (i % 2) * 6.15, y = 1.95 + Math.floor(i / 2) * 2.4, w = 5.95;
  card(s, x, y, w, 2.2, `reason ${i + 1}`);
  circleNum(s, i + 1, x + 0.3, y + 0.3);
  text(s, h, { x: x + 1.05, y: y + 0.28, w: w - 1.3, h: 0.6, fontSize: 20, bold: true, fontFace: THEME.headFontFace, margin: 0, valign: 'middle' });
  text(s, d, { x: x + 1.05, y: y + 0.95, w: w - 1.3, h: 1.1, fontSize: 16, margin: 0 });
});

// ================================================================== 12 limitations and next steps
s = light('Discussion');
s.addText('Limitations and next steps', { placeholder: 'title' });
card(s, M, 1.45, 5.9, 5.1, 'limitations card');
text(s, 'Limitations', { x: M + 0.3, y: 1.6, w: 5.3, h: 0.5, fontSize: 22, bold: true, fontFace: THEME.headFontFace, margin: 0, color: C.accent1 });
text(s, bullets(['One run per model (seed 0)', 'Gaps between models are close to the run-to-run swing', 'One protocol and one dataset',
  'No separate validation set in PRCC']), { x: M + 0.3, y: 2.25, w: 5.4, h: 4.1, fontSize: 17 });
s.addShape(pres.shapes.RECTANGLE, { x: 6.85, y: 1.45, w: 5.88, h: 5.1, fill: { color: C.background1 }, line: { color: 'D8CFCD', width: 1.5 }, objectName: 'next steps card' });
text(s, 'Next steps', { x: 7.15, y: 1.6, w: 5.3, h: 0.5, fontSize: 22, bold: true, fontFace: THEME.headFontFace, margin: 0, color: C.accent1 });
[['1', 'Run seeds 1 and 2, report mean and spread'], ['2', 'Repeat with CAL\'s own settings for all three'], ['3', 'Try the CAL loss on the TransReID backbone']]
  .forEach(([n, t], i) => {
    const y = 2.35 + i * 1.3;
    circleNum(s, n, 7.15, y);
    text(s, t, { x: 7.9, y: y - 0.1, w: 4.6, h: 0.8, fontSize: 17, valign: 'middle', margin: 0 });
  });

// ================================================================== 13 conclusion
pres.addSection({ title: 'Conclusion' });
s = pres.addSlide({ masterName: 'DARK', sectionTitle: 'Conclusion' });
s.addText('Conclusion', { placeholder: 'title' });
s.addText([
  { text: 'Under one protocol, the plain baseline gave the best final result (51.7 mAP).', options: { bullet: true, breakLine: true, paraSpaceAfter: 10 } },
  { text: 'CAL helps right after it switches on, but with these settings the gain does not last.', options: { bullet: true, breakLine: true, paraSpaceAfter: 10 } },
  { text: 'A bigger transformer backbone alone does not solve clothes changes.', options: { bullet: true, breakLine: true, paraSpaceAfter: 10 } },
  { text: 'More seeds are needed before firm conclusions.', options: { bullet: true } },
], { placeholder: 'body' });

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log('written', OUT);
})();
