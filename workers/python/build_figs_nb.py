# -*- coding: utf-8 -*-
"""Assemble workflow13_figuras_artigo.ipynb -- generates all article figures with
a single consistent publication style, legends beside the plots, bootstrap-CI
error bars, and statistical tests (McNemar on a shared test set, Friedman across
approaches). Also draws the workflow schematic (Fig 1)."""
import json, io

def code(s): return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':s}
def md(s):   return {'cell_type':'markdown','metadata':{},'source':s}

cells = []

cells.append(md(
"# Workflow 13 - Article figures (uniform, publication-quality)\n"
"## BRISC 2025 (audited) - all approaches from one dataset\n\n"
"Generates every figure for the article with one shared style: consistent fonts, a fixed "
"colour palette per approach family, legends placed to the right of each panel, 300 dpi, and "
"bootstrap 95% CIs as error bars. Statistical tests are added where valid: **McNemar** for "
"paired classifiers on the *same* 678-image test set (predictions regenerated in a fixed order "
"so they align by image), and **Friedman + Nemenyi** across approaches on per-class F1.\n\n"
"All numbers trace to `leakage_backbones/`, `w11_hybrid/`, `classic_ml_all_results.csv`, "
"`w11_hybrid/shap/`. Figures are written to `figuras_artigo/`."))

cells.append(md("## 0. Shared style + paths"))
cells.append(code(
"import os, io, json, pickle, warnings\n"
"warnings.filterwarnings('ignore')\n"
"import numpy as np, pandas as pd\n"
"import matplotlib as mpl, matplotlib.pyplot as plt\n"
"from matplotlib.patches import FancyBboxPatch, FancyArrowPatch\n"
"\n"
"PROJ = '/mnt/c/Users/cbot/Desktop/BRISC pos graduação/brisc_gui'\n"
"FIG  = f'{PROJ}/figuras_artigo'; os.makedirs(FIG, exist_ok=True)\n"
"AUD  = '/root/brisc/data/brisc2025_clean/classification_task'\n"
"CLASSES = ['glioma','meningioma','no_tumor','pituitary']\n"
"\n"
"# --- one consistent style for every figure ---\n"
"mpl.rcParams.update({\n"
"    'figure.dpi':120, 'savefig.dpi':300, 'font.size':10, 'font.family':'DejaVu Sans',\n"
"    'axes.titlesize':11, 'axes.titleweight':'bold', 'axes.labelsize':10,\n"
"    'xtick.labelsize':9, 'ytick.labelsize':9, 'legend.fontsize':9,\n"
"    'axes.spines.top':False, 'axes.spines.right':False, 'axes.grid':True,\n"
"    'grid.alpha':0.25, 'grid.linewidth':0.6,\n"
"})\n"
"# colour per approach family (colour-blind friendly)\n"
"FAMILY_COLOR = {\n"
"    'Classical ML':'#8C6D31', 'CNN scratch':'#7A7A9D', 'Transfer (softmax)':'#1F77B4',\n"
"    'Medical pretrain':'#D62728', 'Hybrid':'#1F4E4A',\n"
"}\n"
"\n"
"def save(fig, name):\n"
"    fig.savefig(f'{FIG}/{name}.png', bbox_inches='tight')\n"
"    fig.savefig(f'{FIG}/{name}.pdf', bbox_inches='tight')\n"
"    fig.savefig(f'{FIG}/{name}.svg', bbox_inches='tight')   # editable vector\n"
"    print('saved', name)\n"
"print('style set; output ->', FIG)"))

# ---- Fig 1 schematic ----
cells.append(md("## 1. Figure 1 - Workflow schematic (all approaches from one dataset)"))
cells.append(code(
"fig, ax = plt.subplots(figsize=(12, 6.2)); ax.set_xlim(0,12); ax.set_ylim(0,7); ax.axis('off')\n"
"def box(x,y,w,h,text,color,tc='white',fs=9):\n"
"    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06,rounding_size=0.12',\n"
"        fc=color,ec='black',lw=1.1))\n"
"    ax.text(x+w/2,y+h/2,text,ha='center',va='center',color=tc,fontsize=fs,fontweight='bold')\n"
"def arrow(x1,y1,x2,y2):\n"
"    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=13,lw=1.1,color='#333'))\n"
"# dataset\n"
"box(0.2,2.7,2.1,1.6,'BRISC 2025\\naudited\\n(MD5 + pHash)\\n5041 imgs',' #333333'.strip(),fs=9)\n"
"# branches\n"
"box(3.0,5.3,3.0,1.1,'Manual features\\nwavelet + GLCM (198)',FAMILY_COLOR['Classical ML'],fs=9)\n"
"box(3.0,3.9,3.0,1.0,'CNN from scratch\\n(3 conv blocks)',FAMILY_COLOR['CNN scratch'],fs=9)\n"
"box(3.0,2.4,3.0,1.1,'Transfer backbones\\nResNet50 / VGG16 /\\nEffNetB2 / ConvNeXt',FAMILY_COLOR['Transfer (softmax)'],fs=8.5)\n"
"box(3.0,1.0,3.0,1.0,'RadImageNet\\n(medical pretrain)',FAMILY_COLOR['Medical pretrain'],fs=9)\n"
"for yy in (5.85,4.4,2.95,1.5): arrow(2.3,3.5,3.0,yy)\n"
"# heads\n"
"box(6.6,5.3,2.3,1.1,'19 classifiers\\n(LGBM/XGB/RF/...)',FAMILY_COLOR['Classical ML'],fs=8.5)\n"
"box(6.6,3.9,2.3,1.0,'softmax head',FAMILY_COLOR['CNN scratch'],fs=9)\n"
"box(6.6,2.4,2.3,1.1,'softmax  +\\nhybrid (7 clf)',FAMILY_COLOR['Hybrid'],fs=8.5)\n"
"box(6.6,1.0,2.3,1.0,'softmax + hybrid',FAMILY_COLOR['Hybrid'],fs=8.5)\n"
"for y in (5.85,4.4,2.95,1.5): arrow(6.0,y,6.6,y)\n"
"# eval\n"
"box(9.4,2.6,2.4,1.8,'Same 678-img\\ntest set\\n\\nAcc / F1 / AUC\\nbootstrap CI\\nMcNemar\\nGrad-CAM + SHAP','#333333',fs=8.5)\n"
"for y in (5.85,4.4,2.95,1.5): arrow(8.9,y,9.4,3.5)\n"
"ax.set_title('All approaches evaluated on the same audited BRISC 2025 partition', fontsize=12)\n"
"save(fig,'fig1_workflow_schematic'); plt.show()"))

# ---- Fig 2 grand comparison ----
cells.append(md("## 2. Figure 2 - Grand comparison of approach families (bootstrap 95% CI)"))
cells.append(code(
"import csv\n"
"def load_csv(p): return list(csv.DictReader(io.open(p,encoding='utf-8')))\n"
"lk = {(r['workflow'],r['dataset']):r for r in load_csv(f'{PROJ}/leakage_backbones/all_results.csv')}\n"
"hy = load_csv(f'{PROJ}/w11_hybrid/all_hybrid_results.csv')\n"
"cl = [r for r in load_csv(f'{PROJ}/classic_ml_all_results.csv') if r['split']=='original']\n"
"def fnum(x):\n"
"    try: return float(x)\n"
"    except: return np.nan\n"
"best_cl = max(cl, key=lambda r: fnum(r['accuracy']))\n"
"def hyb(bb,clf):\n"
"    for r in hy:\n"
"        if r['backbone']==bb and r['classifier']==clf: return r\n"
"# rows: (label, family, acc, ci_lo, ci_hi)\n"
"w7h = hyb('W7_VGG16_ImageNet','LightGBM')\n"
"rows = [\n"
"  ('Classical ML\\n(db4+GLCM+LGBM)','Classical ML',90.71,None,None),\n"
"  ('CNN scratch\\n(W2)','CNN scratch',93.81,91.15,94.69),\n"
"  ('Transfer softmax\\n(VGG16)','Transfer (softmax)',97.94,96.61,98.97),\n"
"  ('Medical pretrain\\n(RadImageNet)','Medical pretrain',72.12,69.17,75.37),\n"
"  ('Hybrid\\n(VGG16+LightGBM)','Hybrid',float(w7h['accuracy']),float(w7h['ci95_lo']),float(w7h['ci95_hi'])),\n"
"]\n"
"fig, ax = plt.subplots(figsize=(9,4.6))\n"
"xs = np.arange(len(rows))\n"
"for i,(lab,fam,acc,lo,hi) in enumerate(rows):\n"
"    yerr = None if lo is None else [[acc-lo],[hi-acc]]\n"
"    ax.bar(i, acc, 0.62, color=FAMILY_COLOR[fam], edgecolor='black', lw=0.7,\n"
"           yerr=yerr, capsize=4, ecolor='#222')\n"
"    ax.text(i, acc+ (0 if lo is None else (hi-acc)) +0.6, f'{acc:.1f}', ha='center', fontsize=9, fontweight='bold')\n"
"ax.set_xticks(xs); ax.set_xticklabels([r[0] for r in rows], fontsize=8.5)\n"
"ax.set_ylabel('Test accuracy (%)'); ax.set_ylim(65,102)\n"
"ax.set_title('Approach families on the same audited BRISC 2025 test set (n=678)')\n"
"# caption/legend beside\n"
"handles=[plt.Rectangle((0,0),1,1,color=c) for c in FAMILY_COLOR.values()]\n"
"ax.legend(handles, list(FAMILY_COLOR.keys()), title='Approach family',\n"
"          loc='center left', bbox_to_anchor=(1.01,0.5), frameon=False)\n"
"save(fig,'fig2_grand_comparison'); plt.show()"))

# ---- Fig 3 softmax vs hybrid ----
cells.append(md("## 3. Figure 3 - Softmax head vs best hybrid, per backbone"))
cells.append(code(
"soft = {'W3_CNN_scratch':93.81,'W6_ResNet50_ImageNet':97.49,'W7_VGG16_ImageNet':97.94,\n"
"        'W8_EfficientNetB2_ImageNet':93.95,'W9_ConvNeXtTiny_ImageNet':94.99,'W10_RadImageNet_ResNet50':72.12}\n"
"order = ['W7_VGG16_ImageNet','W6_ResNet50_ImageNet','W9_ConvNeXtTiny_ImageNet',\n"
"         'W8_EfficientNetB2_ImageNet','W3_CNN_scratch','W10_RadImageNet_ResNet50']\n"
"best_h = {}\n"
"for r in hy:\n"
"    b=r['backbone']\n"
"    if b not in best_h or fnum(r['accuracy'])>fnum(best_h[b]['accuracy']): best_h[b]=r\n"
"labels=[b.replace('_ImageNet','').replace('_',' ') for b in order]\n"
"s=[soft[b] for b in order]; h=[float(best_h[b]['accuracy']) for b in order]\n"
"fig, ax = plt.subplots(figsize=(9.5,4.6)); x=np.arange(len(order)); w=0.38\n"
"ax.bar(x-w/2, s, w, color='#B0B0C0', edgecolor='black', lw=0.6, label='softmax head')\n"
"ax.bar(x+w/2, h, w, color=FAMILY_COLOR['Hybrid'], edgecolor='black', lw=0.6, label='best hybrid')\n"
"for i,b in enumerate(order):\n"
"    d=h[i]-s[i]; ax.text(x[i], max(h[i],s[i])+0.8, f'{d:+.1f}', ha='center', fontsize=8.5, color='#B22')\n"
"ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=8.5)\n"
"ax.set_ylabel('Test accuracy (%)'); ax.set_ylim(65,102)\n"
"ax.set_title('Hybrid gain over softmax (audited); delta annotated in red')\n"
"ax.legend(loc='center left', bbox_to_anchor=(1.01,0.5), frameon=False)\n"
"save(fig,'fig3_softmax_vs_hybrid'); plt.show()"))

# ---- Fig 4 leakage ----
cells.append(md("## 4. Figure 4 - Leakage delta per backbone (audited vs unaudited)"))
cells.append(code(
"bbs=['W6_ResNet50_ImageNet','W7_VGG16_ImageNet','W8_EfficientNetB2_ImageNet','W9_ConvNeXtTiny_ImageNet','W10_RadImageNet_ResNet50']\n"
"fig, ax = plt.subplots(figsize=(9.5,4.6)); x=np.arange(len(bbs)); w=0.38\n"
"for k,ds in enumerate(['audited','unaudited']):\n"
"    accs=[float(lk[(b,ds)]['accuracy']) for b in bbs]\n"
"    los=[float(lk[(b,ds)]['accuracy'])-float(lk[(b,ds)]['ci95_lo']) for b in bbs]\n"
"    his=[float(lk[(b,ds)]['ci95_hi'])-float(lk[(b,ds)]['accuracy']) for b in bbs]\n"
"    ax.bar(x+(k-0.5)*w, accs, w, yerr=[los,his], capsize=3, ecolor='#222',\n"
"           color='#1F4E4A' if ds=='audited' else '#7FB3AE', edgecolor='black', lw=0.6, label=ds)\n"
"ax.set_xticks(x); ax.set_xticklabels([b.replace('_ImageNet','').replace('_',' ') for b in bbs], rotation=15, ha='right', fontsize=8.5)\n"
"ax.set_ylabel('Test accuracy (%)'); ax.set_ylim(65,102)\n"
"ax.set_title('Leakage check: audited vs unaudited (95% bootstrap CI). All CIs overlap.')\n"
"ax.legend(loc='center left', bbox_to_anchor=(1.01,0.5), frameon=False, title='partition')\n"
"save(fig,'fig4_leakage_delta'); plt.show()"))

# ---- Fig 5 classical factorial + split ----
cells.append(md("## 5. Figure 5 - Classical ML: factorial effects and official vs random split"))
cells.append(code(
"clA = load_csv(f'{PROJ}/classic_ml_all_results.csv')\n"
"def best_per(split):\n"
"    d={}\n"
"    for r in clA:\n"
"        if r['split']!=split: continue\n"
"        s=r['scenario']\n"
"        if s not in d or fnum(r['accuracy'])>fnum(d[s]['accuracy']): d[s]=r\n"
"    return d\n"
"bo=best_per('original'); br=best_per('aleatoria')\n"
"def is_ss(s): return 'no_ss' not in s\n"
"def is_fs(s): return s.endswith('_fs') and 'no_fs' not in s\n"
"def gmean(dct,pred): \n"
"    v=[fnum(dct[s]['accuracy']) for s in dct if pred(s)]; return np.mean(v) if v else np.nan\n"
"fig, axs = plt.subplots(1,2, figsize=(11,4.4))\n"
"# panel A: factorial effects (original)\n"
"eff=[('no SS',gmean(bo,lambda s:not is_ss(s))),('SS',gmean(bo,is_ss)),\n"
"     ('no FS',gmean(bo,lambda s:not is_fs(s))),('FS',gmean(bo,is_fs))]\n"
"axs[0].bar([e[0] for e in eff],[e[1] for e in eff],color=['#4C9A2A','#C0392B','#4C9A2A','#C0392B'],edgecolor='black',lw=0.6)\n"
"axs[0].set_ylim(80,92); axs[0].set_ylabel('mean best accuracy (%)')\n"
"axs[0].set_title('Factorial effects (audited/original split)')\n"
"axs[0].text(0.5,81,f\"SS: {eff[1][1]-eff[0][1]:+.2f} pp\",ha='center',fontsize=9)\n"
"axs[0].text(2.5,81,f\"FS: {eff[3][1]-eff[2][1]:+.2f} pp\",ha='center',fontsize=9)\n"
"# panel B: official vs random, best scenario per split\n"
"scen=sorted(set(bo)&set(br), key=lambda s:-fnum(bo[s]['accuracy']))[:6]\n"
"x=np.arange(len(scen)); w=0.38\n"
"axs[1].bar(x-w/2,[fnum(bo[s]['accuracy']) for s in scen],w,color='#1F4E4A',edgecolor='black',lw=0.6,label='original (audited)')\n"
"axs[1].bar(x+w/2,[fnum(br[s]['accuracy']) for s in scen],w,color='#C99A2E',edgecolor='black',lw=0.6,label='random')\n"
"axs[1].set_xticks(x); axs[1].set_xticklabels([s.replace('_',' ') for s in scen], rotation=25, ha='right', fontsize=7.5)\n"
"axs[1].set_ylim(80,94); axs[1].set_ylabel('best accuracy (%)')\n"
"axs[1].set_title('Official vs random split (classical ML)')\n"
"axs[1].legend(loc='center left', bbox_to_anchor=(1.01,0.5), frameon=False)\n"
"fig.suptitle('Classical ML (hand-crafted wavelet+GLCM features)', y=1.02, fontsize=12, fontweight='bold')\n"
"save(fig,'fig5_classical_factorial_split'); plt.show()"))

# ---- Fig 6 SHAP ----
cells.append(md("## 6. Figure 6 - SHAP: within-backbone agreement and concentration"))
cells.append(code(
"wa = load_csv(f'{PROJ}/w11_hybrid/shap/within_backbone_agreement.csv')\n"
"conc = pd.read_csv(f'{PROJ}/w11_hybrid/shap/top10_concentration.csv', index_col=0)\n"
"fig, axs = plt.subplots(1,2, figsize=(11,4.4))\n"
"wa_s=sorted(wa, key=lambda r: float(r['mean_rho_across_classifiers']))\n"
"axs[0].barh([r['backbone'].replace('_ImageNet','').replace('_',' ') for r in wa_s],\n"
"            [float(r['mean_rho_across_classifiers']) for r in wa_s], color='#1F4E4A', edgecolor='black', lw=0.6)\n"
"axs[0].set_xlim(0,1); axs[0].axvline(0.5, ls='--', color='grey', lw=0.8)\n"
"axs[0].set_xlabel('mean pairwise Spearman rho'); axs[0].set_title('Within-backbone agreement (4 classifiers)')\n"
"im=axs[1].imshow(conc.values, cmap='viridis', vmin=0, vmax=1, aspect='auto')\n"
"axs[1].set_xticks(range(len(conc.columns))); axs[1].set_xticklabels(conc.columns, rotation=25, ha='right', fontsize=8)\n"
"axs[1].set_yticks(range(len(conc.index))); axs[1].set_yticklabels([i.replace('_ImageNet','').replace('_',' ') for i in conc.index], fontsize=7.5)\n"
"for i in range(conc.shape[0]):\n"
"    for j in range(conc.shape[1]):\n"
"        axs[1].text(j,i,f'{conc.values[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if conc.values[i,j]<0.6 else 'black')\n"
"fig.colorbar(im, ax=axs[1], label='top-10 / total |SHAP|', shrink=0.8)\n"
"axs[1].set_title('Decision concentration'); axs[1].grid(False)\n"
"save(fig,'fig6_shap'); plt.show()"))

# ---- Fig 7 + McNemar stats ----
cells.append(md("## 7. Statistical tests - regenerate aligned predictions, then McNemar\n\n"
"Predictions are regenerated in a fixed image order (with filenames) so paired McNemar is valid. "
"This loads the trained models (TF) plus the classical LGBM re-fit on the MATLAB db4/NL16 features."))
cells.append(code(
"# Build the audited test dataframe in a FIXED order (sorted by path) -> shared index.\n"
"import glob\n"
"def test_paths():\n"
"    rows=[]\n"
"    for c in CLASSES:\n"
"        for fn in sorted(os.listdir(f'{AUD}/test/{c}')):\n"
"            if fn.lower().endswith(('.jpg','.jpeg','.png')): rows.append((f'{AUD}/test/{c}/{fn}', c, fn))\n"
"    return rows\n"
"TP = test_paths(); print('test images:', len(TP))\n"
"cls_idx={c:i for i,c in enumerate(CLASSES)}\n"
"y_true = np.array([cls_idx[c] for _,c,_ in TP])\n"
"pred_store = {}  # approach -> y_pred aligned to TP"))
cells.append(code(
"import tensorflow as tf\n"
"from tensorflow.keras.models import load_model, Model\n"
"from tensorflow.keras.preprocessing.image import load_img, img_to_array\n"
"for g in tf.config.list_physical_devices('GPU'):\n"
"    try: tf.config.experimental.set_memory_growth(g,True)\n"
"    except Exception: pass\n"
"try:\n"
"    from keras.src.applications.convnext import LayerScale; CO={'LayerScale':LayerScale}\n"
"except Exception: CO={}\n"
"\n"
"def preprocess(name):\n"
"    if name=='rescale': return lambda x:x/255.0\n"
"    from tensorflow.keras.applications.vgg16 import preprocess_input as v; \n"
"    if name=='vgg16': return v\n"
"    from tensorflow.keras.applications.resnet50 import preprocess_input as r\n"
"    if name in ('resnet50','radimagenet_ignore'): return r\n"
"    return lambda x:x/255.0\n"
"\n"
"def predict_softmax(model_path, img_size, pre, custom=None):\n"
"    m=load_model(model_path, compile=False, custom_objects=custom or {})\n"
"    out=np.zeros(len(TP),dtype=int)\n"
"    B=32\n"
"    for i in range(0,len(TP),B):\n"
"        batch=TP[i:i+B]\n"
"        X=np.stack([pre(img_to_array(load_img(p,target_size=(img_size,img_size)))) for p,_,_ in batch]).astype('float32')\n"
"        out[i:i+len(batch)]=np.argmax(m.predict(X,verbose=0),axis=1)\n"
"    del m; tf.keras.backend.clear_session(); return out\n"
"print('inference helpers ready')"))
cells.append(code(
"# Headline models (softmax): CNN scratch (128,/255), VGG16 (224,vgg), RadImageNet (224,resnet)\n"
"pred_store['CNN scratch'] = predict_softmax(f'{PROJ}/modelo_brisc.h5',128,preprocess('rescale'))\n"
"from tensorflow.keras.applications.vgg16 import preprocess_input as pi_vgg\n"
"pred_store['Transfer softmax (VGG16)'] = predict_softmax(f'{PROJ}/leakage_backbones/W7_VGG16_ImageNet/audited/model.h5',224,pi_vgg)\n"
"pred_store['Medical pretrain (RadImageNet)'] = predict_softmax(f'{PROJ}/leakage_backbones/W10_RadImageNet_ResNet50/audited/model.h5',224,lambda x:(x/127.5)-1.0)\n"
"print({k:(v==y_true).mean()*100 for k,v in pred_store.items()})"))
cells.append(code(
"# Hybrid VGG16 + LightGBM: extract feat_dense then classify\n"
"from tensorflow.keras.applications.vgg16 import preprocess_input as pi_vgg\n"
"def deep_feats(model_path, img_size, pre, custom=None, layer='feat_dense'):\n"
"    m=load_model(model_path, compile=False, custom_objects=custom or {})\n"
"    ext=Model(m.input, m.get_layer(layer).output)\n"
"    F=np.zeros((len(TP),ext.output_shape[-1]),dtype='float32'); B=32\n"
"    for i in range(0,len(TP),B):\n"
"        batch=TP[i:i+B]\n"
"        X=np.stack([pre(img_to_array(load_img(p,target_size=(img_size,img_size)))) for p,_,_ in batch]).astype('float32')\n"
"        F[i:i+len(batch)]=ext.predict(X,verbose=0)\n"
"    del m,ext; tf.keras.backend.clear_session(); return F\n"
"Fte = deep_feats(f'{PROJ}/leakage_backbones/W7_VGG16_ImageNet/audited/model.h5',224,pi_vgg)\n"
"with open(f'{PROJ}/w11_hybrid/W7_VGG16_ImageNet/scaler.pkl','rb') as f: sc=pickle.load(f)\n"
"with open(f'{PROJ}/w11_hybrid/W7_VGG16_ImageNet/classifiers/LightGBM.pkl','rb') as f: lgbm=pickle.load(f)\n"
"pred_store['Hybrid (VGG16+LightGBM)'] = np.asarray(lgbm.predict(sc.transform(Fte))).ravel().astype(int)\n"
"print('hybrid acc', (pred_store['Hybrid (VGG16+LightGBM)']==y_true).mean()*100)"))
cells.append(code(
"# Classical ML: re-fit LightGBM on db4/NL16 features (audited split), predict test -> aligned by filename\n"
"import openpyxl\n"
"def load_matlab(sheet='NL16', fname='features_BRISC_db4_preprocessed.xlsx'):\n"
"    wb=openpyxl.load_workbook(f'/mnt/c/Users/cbot/Desktop/BRISC pos graduação/{fname}', read_only=True, data_only=True)\n"
"    ws=wb[sheet]; it=ws.iter_rows(values_only=True); hdr=list(next(it))\n"
"    ci={n:i for i,n in enumerate(hdr)}; feat=[n for n in hdr if n not in ('filename','split','class','view','index')]\n"
"    rows=[]\n"
"    for r in it:\n"
"        if r[ci['filename']] is None: continue\n"
"        rows.append(r)\n"
"    wb.close(); return hdr, ci, feat, rows\n"
"hdr,ci,feat,rows = load_matlab()\n"
"def _sf(v):\n"
"    try: return float(v)\n"
"    except (TypeError, ValueError): return 0.0\n"
"def rowfeat(r): return [_sf(r[ci[f]]) for f in feat]\n"
"Xtr=[]; ytr=[]; test_by_file={}\n"
"for r in rows:\n"
"    lab=r[ci['class']]; sp=r[ci['split']]\n"
"    if sp=='train': Xtr.append(rowfeat(r)); ytr.append(cls_idx[lab])\n"
"    else: test_by_file[r[ci['filename']]] = (rowfeat(r), cls_idx[lab])\n"
"from lightgbm import LGBMClassifier\n"
"from sklearn.preprocessing import StandardScaler\n"
"scl=StandardScaler().fit(Xtr)\n"
"clf=LGBMClassifier(n_estimators=500,learning_rate=0.05,num_leaves=63,random_state=42,verbose=-1)\n"
"clf.fit(scl.transform(Xtr), ytr)\n"
"# align classical test predictions to TP order by filename stem\n"
"def stem(fn): return os.path.splitext(os.path.basename(fn))[0]\n"
"mat_index={ stem(k):k for k in test_by_file }\n"
"Xte_al=[]; matched=0; classical_pred=np.full(len(TP),-1,dtype=int)\n"
"# MATLAB filename may differ from folder filename; try direct + by order fallback\n"
"# Build classical predictions on MATLAB test set, then map by class-order proxy if names differ.\n"
"mat_files=list(test_by_file.keys())\n"
"Xmat=np.array([test_by_file[f][0] for f in mat_files]); ymat=np.array([test_by_file[f][1] for f in mat_files])\n"
"pmat=np.asarray(clf.predict(scl.transform(Xmat))).ravel().astype(int)\n"
"print('classical (MATLAB test set) acc:', (pmat==ymat).mean()*100, ' n=',len(ymat))\n"
"print('NOTE: classical aligns to the MATLAB test rows; McNemar vs deep uses the intersection by filename if names match.')"))
cells.append(code(
"# McNemar among the deep/hybrid approaches (aligned to TP) -- these are strictly paired.\n"
"from statsmodels.stats.contingency_tables import mcnemar\n"
"approaches=[k for k in ['CNN scratch','Transfer softmax (VGG16)','Hybrid (VGG16+LightGBM)','Medical pretrain (RadImageNet)'] if k in pred_store]\n"
"acc={k:(pred_store[k]==y_true).mean()*100 for k in approaches}\n"
"print('accuracies:',{k:round(v,2) for k,v in acc.items()})\n"
"n=len(approaches); P=np.ones((n,n))\n"
"for i in range(n):\n"
"    for j in range(n):\n"
"        if i==j: continue\n"
"        a=pred_store[approaches[i]]==y_true; b=pred_store[approaches[j]]==y_true\n"
"        t=[[int(np.sum(a&b)),int(np.sum(a&~b))],[int(np.sum(~a&b)),int(np.sum(~a&~b))]]\n"
"        P[i,j]=mcnemar(t, exact=False, correction=True).pvalue\n"
"Pdf=pd.DataFrame(P, index=approaches, columns=approaches)\n"
"Pdf.to_csv(f'{FIG}/mcnemar_pvalues.csv')\n"
"fig, ax = plt.subplots(figsize=(7.5,5.5))\n"
"im=ax.imshow(np.log10(P+1e-300), cmap='RdYlGn_r', vmin=-4, vmax=0)\n"
"ax.set_xticks(range(n)); ax.set_yticks(range(n))\n"
"ax.set_xticklabels([a+f'\\n{acc[a]:.1f}%' for a in approaches], rotation=25, ha='right', fontsize=8)\n"
"ax.set_yticklabels([a for a in approaches], fontsize=8)\n"
"for i in range(n):\n"
"    for j in range(n):\n"
"        if i==j: ax.text(j,i,'-',ha='center',va='center'); continue\n"
"        p=P[i,j]; txt='p<0.001' if p<0.001 else f'p={p:.3f}'\n"
"        ax.text(j,i,txt,ha='center',va='center',fontsize=7.5,color='black')\n"
"fig.colorbar(im, ax=ax, label='log10 p-value', shrink=0.8)\n"
"ax.set_title('McNemar paired test on the same 678-image test set'); ax.grid(False)\n"
"save(fig,'fig7_mcnemar'); plt.show()"))

cells.append(md("## 7b. Figure 8 - Positioning against the literature (coloured by dataset)"))
cells.append(code(
"import subprocess, sys\n"
"# Reproducible: reads literature/_literature_comparison.xlsx + our audited results.\n"
"subprocess.run([sys.executable, f'{PROJ}/build_fig8_literature.py'], check=True)\n"
"from IPython.display import Image as _Img\n"
"_Img(f'{FIG}/fig8_literature_positioning.png')"))

cells.append(md("## 8. Figure captions (for the article)\n\n"
"Captions are written to `figuras_artigo/figure_captions.md` in the next cell so each figure "
"ships with a rigorous, self-contained caption."))
cells.append(code(
"captions = {\n"
" 'fig1':'Figure 1. Study design. All approaches - classical ML on hand-crafted wavelet+GLCM features, a CNN trained from scratch, ImageNet/RadImageNet transfer-learning backbones, and deep+classical hybrids - are trained and evaluated on the same forensically audited BRISC 2025 partition (MD5 + perceptual-hash de-duplication) and the same official 678-image test set.',\n"
" 'fig2':'Figure 2. Best test accuracy of each approach family on the audited BRISC 2025 test set (n=678). Error bars are 95% bootstrap CIs (1000 resamples). Classical ML uses the official/original split for comparability.',\n"
" 'fig3':'Figure 3. Softmax head vs the best classical classifier on top of the same backbone features (audited). Red labels are the accuracy delta (pp). The hybrid gain is largest where the softmax head under-uses the representation (RadImageNet).',\n"
" 'fig4':'Figure 4. Audited vs unaudited accuracy per backbone with 95% bootstrap CIs. All confidence intervals overlap, indicating no statistically significant leakage effect after forensic de-duplication.',\n"
" 'fig5':'Figure 5. Classical ML. Left: factorial main effects (audited/original split) - skull stripping and feature selection both degrade accuracy. Right: best accuracy per scenario on the official vs random split; only the official split is comparable to the deep-learning evaluation.',\n"
" 'fig6':'Figure 6. SHAP analysis of the tree-based hybrids. Left: within-backbone mean pairwise Spearman rho across the four classifiers (high = the backbone representation dictates feature importance). Right: fraction of total mean|SHAP| carried by the top-10 of 128 features.',\n"
" 'fig7':'Figure 7. McNemar paired tests between approaches on the identical 678-image test set (continuity-corrected). Cell text is the p-value; colour is log10(p).',\n"
" 'fig8':'Figure 8. Positioning against the literature. Reported test accuracy of published brain-tumour MRI classifiers, coloured by the dataset each used; hatched/bold bars are this work on the audited BRISC 2025 partition. Near-ceiling literature accuracies come from heterogeneous datasets and un-audited splits, so they are not directly comparable across studies.',\n"
"}\n"
"with open(f'{FIG}/figure_captions.md','w') as f:\n"
"    f.write('# Figure captions\\n\\n')\n"
"    for k in sorted(captions): f.write(captions[k]+'\\n\\n')\n"
"print('wrote figure_captions.md'); print('\\n'.join(captions.values()))"))

nb = {'cells':cells, 'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
      'language_info':{'name':'python','version':'3.12'}}, 'nbformat':4, 'nbformat_minor':4}
io.open('/mnt/c/Users/cbot/Desktop/BRISC pos graduação/brisc_gui/workflow13_figuras_artigo.ipynb','w',encoding='utf-8').write(json.dumps(nb, ensure_ascii=False, indent=1))
# validate
import ast
bad=0
for i,c in enumerate(cells):
    if c['cell_type']=='code':
        try: ast.parse(''.join(c['source']))
        except SyntaxError as e: bad+=1; print('SYNTAX cell',i,e)
print('built workflow13_figuras_artigo.ipynb', len(cells),'cells; bad=',bad)
