"""Generate a corrected, explicitly scoped comparison figure from recomputed tables."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics',type=Path,default=ROOT/'results/corrected/metrics.csv')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/figures')
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a fresh output directory')
    frame=pd.read_csv(args.metrics).set_index('model')
    keys=['cnn_canonical','vgg16','vgg16_lgbm','radimagenet']
    labels=['Canonical CNN','VGG16 softmax','VGG16 + LightGBM*','RadImageNet ResNet50']
    selected=frame.loc[keys]
    fig,ax=plt.subplots(figsize=(8.4,4.7),layout='constrained')
    colors=['#777777','#503868','#936caf','#b6b6b6']
    bars=ax.barh(labels,selected.accuracy,color=colors)
    ax.errorbar(selected.accuracy,labels,
                xerr=[selected.accuracy-selected.ci95_lo,selected.ci95_hi-selected.accuracy],
                fmt='none',color='#222222',capsize=4,lw=1.1)
    for bar,(_,row) in zip(bars,selected.iterrows()):
        ax.text(101,bar.get_y()+bar.get_height()/2,f'{row.accuracy:.2f}% [{row.ci95_lo:.2f}, {row.ci95_hi:.2f}]',va='center',fontsize=9)
    ax.set_xlim(0,135);ax.set_xticks([0,25,50,75,100]);ax.invert_yaxis()
    ax.set_xlabel('Accuracy (%) — class-stratified bootstrap 95% CI')
    ax.set_title('Verified inference on the historical BRISC 2025 test cohort (n=678)',fontsize=12)
    ax.spines[['top','right']].set_visible(False)
    fig.get_layout_engine().set(rect=(0,.15,1,.83))
    fig.text(.02,.025,'*Post hoc best-of-seven test selection. CIs are conditional on fixed models.\nHistorical pHash curation remains under review; patient-level independence is unverified.',fontsize=9)
    args.output.mkdir(parents=True)
    for extension in ['png','svg','pdf']:fig.savefig(args.output/f'verified_comparison.{extension}',dpi=200)
    print('Saved:',args.output)

if __name__=='__main__':main()
